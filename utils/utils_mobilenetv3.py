import tensorrt as trt
from cuda import cudart
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import json
import os
from utils import common 

class BaseEngine(object):
    def __init__(self, engine_path):

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            size = np.prod(shape) * np.dtype(trt.nptype(dtype)).itemsize
            allocation = common.cuda_call(cudart.cudaMalloc(size))

            binding = {
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': shape,
                'allocation': allocation
            }
            self.allocations.append(allocation)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, batch_images):
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # Record start time
        start_time = time.time()

        # Transfer input data to GPU
        common.memcpy_host_to_device(self.inputs[0]['allocation'], np.ascontiguousarray(batch_images))

        # Execute the network
        self.context.execute_v2(self.allocations)

        # Retrieve output from GPU
        output = np.empty(self.outputs[0]['shape'], dtype=self.outputs[0]['dtype'])
        common.memcpy_device_to_host(output, self.outputs[0]['allocation'])

        # Record end time
        end_time = time.time()

        # Calculate and print inference time for this frame
        inference_time = end_time - start_time
        print(f"Inference time for current frame: {inference_time:.4f} seconds")
        
        return output

