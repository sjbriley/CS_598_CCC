import time
import torch
import grpc
import data_feed_pb2
import data_feed_pb2_grpc
from torchvision import datasets, transforms
import numpy as np
import zlib
import time
from decision_engine import DecisionEngine 
import sys
from io import BytesIO
from PIL import Image
import os
from utils import DecodeJPEG, ConditionalNormalize

class Profiler:
    def __init__(self, batch_size, dataset_path, grpc_host, grpc_port):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        # Automatically choose MPS (for macOS GPU) if available, otherwise CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def stage_one_profiling(self):
        # 1. Measure GPU throughput with synthetic data (remains the same)
        num_samples_gpu = self.batch_size * 50
        start = time.time()
        dummy_data = torch.randn(self.batch_size, 3, 224, 224).to(self.device)
        for _ in range(50):
            _ = dummy_data + dummy_data  # Dummy operation
        gpu_time = time.time() - start
        gpu_throughput = num_samples_gpu / gpu_time

        # 2. Measure I/O throughput over gRPC (between training and storage node)
        num_samples_io = 0
        start = time.time()
        channel = grpc.insecure_channel(f'{self.grpc_host}:{self.grpc_port}')
        stub = data_feed_pb2_grpc.DataFeedStub(channel)
        sample_stream = stub.StreamSamples(iter([]))  # Empty iterator

        # Simulate fetching 50 batches to measure I/O throughput
        for i, sample_batch in enumerate(sample_stream):
            if i >= 50:
                break
            for sample in sample_batch.samples:
                num_samples_io += 1
                
                # If the sample is an image (serialized as bytes), convert it back to PIL image
                if isinstance(sample.image, bytes):
                    image_pil = Image.open(BytesIO(sample.image))  # Deserialize the bytes to PIL image

                elif isinstance(sample.image, torch.Tensor):
                    # If it's already a tensor, handle it as a tensor
                    image_tensor = sample.image  # Tensors received directly

                # For throughput measurement, you can skip converting to tensor at this point
                # Measure I/O throughput based on the received samples.

        io_time = time.time() - start
        io_throughput = num_samples_io / io_time

        # 3. Measure CPU throughput (local preprocessing)
        num_samples_cpu = 0
        start = time.time()
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(self.dataset_path, transform=None)
        for i, (data, _) in enumerate(dataset):
            if i >= 50:
                break
            # Apply transformations to the PIL image
            transformed_image = transform(data)
            num_samples_cpu += 1  # Increment by the number of samples processed

        cpu_time = time.time() - start
        cpu_throughput = num_samples_cpu / cpu_time

        return gpu_throughput, io_throughput, cpu_throughput

    def stage_two_profiling(self):
        cpu_device = torch.device("cpu")
        channel = grpc.insecure_channel(f'{self.grpc_host}:{self.grpc_port}')
        stub = data_feed_pb2_grpc.DataFeedStub(channel)
        samples = stub.StreamSamples(iter([]))

        sample_metrics = []
        for i, sample_batch in enumerate(samples):
            if i >= 50:
                break

            for sample in sample_batch.samples:
                # If the sample is serialized as bytes (PIL image), deserialize it
                if isinstance(sample.image, bytes):
            
                    original_size = len(sample.image)
                    # Preprocess the sample
                    transformed_data, times_per_transformation, transformed_sizes_per_transformation = self.preprocess_sample(sample.image)
                elif isinstance(sample.image, torch.Tensor):
                    # If it's already a tensor, process the tensor
                    image_tensor = sample.image
                    original_size = image_tensor.nelement() * image_tensor.element_size()  # Calculate size of tensor
                    # Preprocess the sample (tensor)
                    transformed_data, times_per_transformation, transformed_sizes_per_transformation = self.preprocess_sample(image_tensor)

                # Append the metrics
                sample_metrics.append({
                    'original_size': original_size,
                    'transformed_sizes': transformed_sizes_per_transformation,
                    'preprocessing_times': times_per_transformation
                })

        return sample_metrics

    def preprocess_sample(self, sample):
    # List of transformations to apply individually
        decode_jpeg = self.DecodeJPEG()
        
        transformations = [
            decode_jpeg,  # Decode raw JPEG bytes to a PIL image
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Converts PIL images to tensors
            ConditionalNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Conditional normalization
        ]

        processed_sample = sample
        sizes = []
        times = []
        
        # Apply each transformation individually and time it
        for transform in transformations:
            if transform is not None:
                start_time = time.time()
                processed_sample = transform(processed_sample)  # Apply transformation
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)  # Record the time for each transformation
            else:
                times.append(0)  # No time taken for ToTensor() since it's a no-op
            if isinstance(processed_sample, torch.Tensor):
            # For PyTorch tensors
                data_size = processed_sample.nelement() * processed_sample.element_size()
            elif isinstance(processed_sample, np.ndarray):
                # For NumPy arrays
                data_size = processed_sample.nbytes
            elif isinstance(processed_sample, Image.Image):
                # For PIL images
                data_size = len(processed_sample.tobytes())  # Convert to bytes and measure length
            else:
                # Fallback to sys.getsizeof for other types
                data_size = sys.getsizeof(processed_sample)

            sizes.append(data_size)

        return processed_sample, times, sizes
    
    

    def run_profiling(self):
        # Stage 1: Basic throughput analysis
        gpu_throughput, io_throughput, cpu_preprocessing_throughput = self.stage_one_profiling()
        print("GPU Throughput:", gpu_throughput)
        print("I/O Throughput:", io_throughput)
        print("CPU Preprocessing Throughput:", cpu_preprocessing_throughput)
        if io_throughput < cpu_preprocessing_throughput:
            # Stage 2: Detailed sample-specific profiling
            return gpu_throughput, io_throughput, cpu_preprocessing_throughput, self.stage_two_profiling()
        return gpu_throughput, io_throughput, cpu_preprocessing_throughput, None
    



if __name__ == '__main__':
    profiler = Profiler(batch_size=2, dataset_path='imagenet', grpc_host='localhost', grpc_port=50051)
    gpu_throughput, io_throughput, cpu_preprocessing_throughput, sample_metrics = profiler.run_profiling()
    if sample_metrics:  # If the profiler identifies an I/O bottleneck
        decision_engine = DecisionEngine(sample_metrics, gpu_t = gpu_throughput, cpu_t =cpu_preprocessing_throughput, io_t = io_throughput,  cpu_cores_compute=1, cpu_cores_storage=8, grpc_host='localhost', grpc_port=50051)
        offloading_plan = decision_engine.iterative_offloading()
        print(offloading_plan)

