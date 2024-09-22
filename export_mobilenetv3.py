import os
import sys
import logging
import argparse
import numpy as np
import tensorrt as trt
from cuda import cudart
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from utils import common 
from DataLoaderBatcher_mobilenetv3 import DataLoaderBatcher

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, batch_size=16, max_batches=100):
        """
        Initializes the EngineCalibrator with an internal DataLoaderBatcher for CIFAR-10 test set.
        :param cache_file: The file where calibration cache will be stored.
        :param batch_size: The size of the batches used for calibration.
        :param max_batches: Maximum number of batches to use for calibration.
        """
        super().__init__()
        self.cache_file = cache_file
        self.batch_allocation = None
        self.batch_generator = None
        self.batch_size = batch_size
        self.max_batches = max_batches

        # Internal method to setup the CIFAR-10 DataLoader
        if not os.path.exists(self.cache_file):
            self._prepare_dataloader()


    def _prepare_dataloader(self):
        """
        Prepares the internal DataLoader for the CIFAR-10 test dataset and wraps it in a DataLoaderBatcher.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to MobileNetV3 input shape
            transforms.ToTensor(),
        ])

        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Set up the DataLoaderBatcher using the test_loader
        batch_shape = [self.batch_size, 3, 224, 224]  # [batch_size, channels, height, width]
        self.batcher = DataLoaderBatcher(test_loader, shape=batch_shape, dtype=np.float32, max_batches=self.max_batches)

        # Set the batch generator for calibration
        self.batch_generator = self.batcher.get_batch()
        size = int(np.dtype(np.float32).itemsize * np.prod(batch_shape))
        self.batch_allocation = common.cuda_call(cudart.cudaMalloc(size))


    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        :return: Batch size used during calibration.
        """
        return self.batch_size

    def get_batch(self, names):
        """
        Fetches the next batch from the DataLoaderBatcher for calibration.
        :param names: Tensor names (not used in this case).
        :return: A list of device pointers (one per input tensor).
        """
        try:
            batch, _, _ = next(self.batch_generator)
            log.info("Calibrating image {} / {}".format(self.batcher.batch_index, len(self.batcher.dataloader.dataset) // self.batcher.batch_size))
            common.memcpy_host_to_device(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Reads the calibration cache if it exists.
        """
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Writes the calibration cache to the file.
        """
        if self.cache_file:
            with open(self.cache_file, "wb") as f:
                log.info("Writing calibration cache data to: {}".format(self.cache_file))
                f.write(cache)

class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """
    def __init__(self, verbose=False, workspace=8):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (2 ** 30))
        # self.config.max_workspace_size = workspace * (2 ** 30)  # Deprecation

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                print("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    print(self.parser.get_error(error))
                sys.exit(1)

        # Log the network structure
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]
        print("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            print("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            print("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))

        assert self.batch_size > 0


    def create_engine(self, engine_path, precision, calib_input=None, calib_cache=None, calib_num_images=5000,
                      calib_batch_size=8):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images (if INT8).
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration (if INT8).
        :param calib_batch_size: The batch size to use for the calibration process.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        print("Building {} Engine in {}".format(precision, engine_path))
        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                print("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                print("INT8 is not supported natively on this platform/device")
            else:
                if self.builder.platform_has_fast_fp16:
                    self.config.set_flag(trt.BuilderFlag.FP16)
                self.config.set_flag(trt.BuilderFlag.INT8)
                self.config.int8_calibrator = EngineCalibrator(calib_cache)
                if not os.path.exists(calib_cache):
                    print(f"Starting INT8 Calibration with CIFAR-10 test set.")

        with self.builder.build_serialized_network(self.network, self.config) as engine, open(engine_path, "wb") as f:
            print("Serializing engine to file: {:}".format(engine_path))
            f.write(engine)

def main(args):
    builder = EngineBuilder(args.verbose, args.workspace)
    builder.create_network(args.onnx)
    builder.create_engine(args.engine, args.precision, args.calib_input, args.calib_cache, args.calib_num_images,
                          args.calib_batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "int8"],
                        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument("-w", "--workspace", default=1, type=int, help="The max memory workspace size to allow in Gb, "
                                                                       "default: 1")
    parser.add_argument("--calib_input", help="The directory holding images to use for calibration")
    parser.add_argument("--calib_cache", default="./calibration.cache",
                        help="The file path for INT8 calibration cache to use, default: ./calibration.cache")
    parser.add_argument("--calib_num_images", default=5000, type=int,
                        help="The maximum number of images to use for calibration, default: 5000")
    parser.add_argument("--calib_batch_size", default=8, type=int,
                        help="The batch size for the calibration process, default: 8")

    args = parser.parse_args()
    print(args)
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)
    if args.precision == "int8" and not (args.calib_input or os.path.exists(args.calib_cache)):
        parser.print_help()
        log.error("When building in int8 precision, --calib_input or an existing --calib_cache file is required")
        sys.exit(1)
    
    main(args)


