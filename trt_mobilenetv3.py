from utils.utils_mobilenetv3 import BaseEngine
import numpy as np
import cv2
import time
import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms


class MobileNetV3Predictor(BaseEngine):
    def __init__(self, engine_path, device):
        super(MobileNetV3Predictor, self).__init__(engine_path)
        self.device = device
        self.n_classes = 10  # CIFAR-10


    def evaluate(self, dataloader):
        correct = 0
        total = 0
        inference_time = 0
        for images, labels in dataloader:
            images = images.numpy()  # Convert to NumPy array

            start_time = time.time()
            outputs = self.infer(images)
            end_time = time.time()

            # Inference time tracking
            inference_time += (end_time - start_time)

            # Compute predictions
            predictions = np.argmax(outputs, axis=1)
            correct += np.sum(predictions == labels.numpy())
            total += labels.size(0)

        accuracy = correct / total
        avg_inference_time = inference_time / len(dataloader)
        return avg_inference_time, accuracy
    

def load_cifar10_data():
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    return test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path")
    parser.add_argument("-i", "--image", help="image path")
    parser.add_argument("-o", "--output", help="image output path")
    parser.add_argument("-v", "--video",  help="video path or camera index ")
    parser.add_argument("-t", "--test",  help="calculate mAP on COCO dataset")

    args = parser.parse_args()
    print(args)

    pred = MobileNetV3Predictor(engine_path=args.engine, device="cuda")
    test_loader = load_cifar10_data()
    avg_inference_time, accuracy = pred.evaluate(test_loader)

    print(f"Average Inference Time: {avg_inference_time:.4f} seconds")
    print(f"Test Accuracy: {accuracy:.4f}")

    # img_path = args.image
    # video = args.video
    # test = args.test
    # if img_path:
    #   origin_img = pred.inference(img_path, conf=0.1, end2end=args.end2end)

    #   cv2.imwrite("%s" %args.output , origin_img)
    # if video:
    #   pred.detect_video(video, conf=0.1, end2end=args.end2end) # set 0 use a webcam
    # if test:
    #   pred.test()