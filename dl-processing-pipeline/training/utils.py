from PIL import Image
from io import BytesIO
from torchvision import transforms
import torch
import grpc
import data_feed_pb2
import data_feed_pb2_grpc
import numpy as np
import zlib
import torch.utils.data
import os
from torch.utils.data import Dataset, DataLoader
import time

import cProfile
import pstats
import io



class DecodeJPEG:
        def __call__(self, raw_bytes):
            return Image.open(BytesIO(raw_bytes))
        
class ConditionalNormalize:
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, tensor):
        # Only apply normalization if the tensor has 3 channels
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)  # Repeat the single channel across the 3 RGB channels
        
        # Apply normalization to 3-channel (RGB) images
        return self.normalize(tensor)

class RemoteDataset(torch.utils.data.IterableDataset):
    def __init__(self, host, port, batch_size=256):
        self.host = host
        self.port = port
        self.batch_size = batch_size
        print(f"Initialized RemoteDataset with host={self.host}, port={self.port}, batch_size={self.batch_size}")

    def __iter__(self):
        print("Starting RemoteDataset __iter__")
        
        
        try:
            connect_start = time.time()
            channel = grpc.insecure_channel(
                f'{self.host}:{self.port}',
                options=[
                    ('grpc.max_send_message_length', -1),  # 64 MB
                    ('grpc.max_receive_message_length', -1),  # 64 MB
                    ('grpc.http2.max_pings_without_data', 0),  # No limit
                    ('grpc.http2.min_time_between_pings_ms', 10000),
                    ('grpc.http2.min_ping_interval_without_data_ms', 10000),
                    ('grpc.http2.max_frame_size', 16777216),  # 16 MB, adjust as needed
                ]
            )

            stub = data_feed_pb2_grpc.DataFeedStub(channel)
            config_request = data_feed_pb2.Config(batch_size=self.batch_size)
            sample_stream = stub.get_samples(config_request)
            batch_start = time.time()
            batch_time = 0
            

            # print("Requesting data with batch size:", self.batch_size)
            
            batch_images = []
            batch_labels = []
            
            for sample_batch in sample_stream:
                for sample in sample_batch.samples:
                    decompress_start = time.time()
                    # Deserialize image data
                    img_data = sample.image
                    if sample.is_compressed:
                        img_data = zlib.decompress(img_data)
                    decompress_end = time.time()
                    # print(f"Decompression time: {decompress_end - decompress_start:.4f} seconds")
                    # Convert img_data to tensor and add to batch
                    img_tensor = self.preprocess_sample(img_data, sample.transformations_applied)

                    # print(f"Transformation time: {transform_end - transform_start:.4f} seconds")

                    batch_images.append(img_tensor)
                    batch_labels.append(torch.tensor(sample.label))

                    # Yield a batch when it reaches the desired batch size
                    if len(batch_images) == self.batch_size:
                        yield torch.stack(batch_images), torch.stack(batch_labels)
                        batch_end = time.time()
                        batch_time = batch_end - batch_start
                        batch_start = time.time()
                        # print(f"Yielded a batch of size: {self.batch_size} in {batch_time:.4f} seconds")
                        batch_images = []
                        batch_labels = []
                        # print(f"Yielded a batch of size: {self.batch_size}")

        except Exception as e:
            print(f"Unexpected error in RemoteDataset __iter__: {e}")


    def preprocess_sample(self, sample, transformations_applied):
        try:
            if 0 < transformations_applied <= 3:
                    sample = Image.open(BytesIO(sample)).convert('RGB')
            elif transformations_applied > 3:
                img_array = np.frombuffer(sample, dtype=np.float32).copy()
                # Reshape based on expected image shape, e.g., (3, 224, 224) for RGB images
                sample = torch.from_numpy(img_array.reshape(3, 224, 224))
            # List of transformations to apply individually
            decode_jpeg = DecodeJPEG()
            
            
            transformations = [
                decode_jpeg,  # Decode raw JPEG bytes to a PIL image
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Converts PIL images to tensors
                ConditionalNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Conditional normalization
            ]

            processed_sample = sample
            for i in range(transformations_applied, len(transformations)):
                if transformations[i] is not None:
                    processed_sample = transformations[i](processed_sample)
        except Exception as e:
            print(f"Error in preprocess_sample: {e}")
            return None
        return processed_sample

class ImagePathDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.targets = []

        # Traverse the directory structure and collect image paths and targets
        for class_idx, class_name in enumerate(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.targets.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        target = self.targets[idx]
        
        return img_path, target  # Only return two values: path and target
def custom_collate_fn(batch):
    images = []
    labels = []
    
    for img, label in batch:
        images.append(img)
        labels.append(label)
        
    # Stack images and labels into a batch
    images = torch.stack(images)
    labels = torch.stack(labels)
    
    return images, labels

    
if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    # Example usage of RemoteDataset
    dataset = RemoteDataset(host='localhost', port=50051, batch_size=32)
    train_loader = DataLoader(
        dataset, batch_size=None, num_workers=0, pin_memory=False
    )

    total_images = 0
    target_images = 1000
    start_time = time.time()
    batch_time_start = time.time()
    for i, (images, target) in enumerate(train_loader):
        
        batch_size = images.shape[0]  # Current batch size (could vary depending on availability)
        total_images += batch_size

        # Flatten the nested batches into a single batch dimension
        # print(f"Batch {i}: Loaded {batch_size} images. Total loaded so far: {total_images}")
        images = images.view(-1, 3, 224, 224)  # Flatten: (2, 2, 3, 224, 224) -> (4, 3, 224, 224)
        target = target.view(-1)  # Adjust target as well
        batch_time_end = time.time()
        # print(f"Batch {i}: Loaded {batch_size} images in {batch_time_end - batch_time_start:.4f} seconds")
        batch_time_start = time.time()

        # Stop if we've loaded 1000 images
        if total_images >= target_images:
            end_time = time.time()
            elapsed_time = end_time - start_time
            throughput = total_images / elapsed_time  # Images per second

            print(f"Loaded {total_images} images in {elapsed_time:.2f} seconds. Throughput: {throughput:.2f} images/second")
            # profiler.disable()
            # stream = io.StringIO()
            # stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
            # stats.print_stats(20)  # Display top 20 time-consuming functions
            # print(stream.getvalue())
            break
    

    # Measure end time and calculate throughput
    
