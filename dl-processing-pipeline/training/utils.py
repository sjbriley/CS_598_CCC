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
from torch.utils.data import Dataset


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

    def __iter__(self):
        channel = grpc.insecure_channel(
            f'{self.host}:{self.port}',
            options=[
                ('grpc.max_send_message_length', 800 * 1024 * 1024),  # 800 MB
                ('grpc.max_receive_message_length', 800 * 1024 * 1024)  # 800 MB
            ]
        )
        stub = data_feed_pb2_grpc.DataFeedStub(channel)
    
        samples = stub.StreamSamples(iter([]))
        
        for i, sample_batch in enumerate(samples):

            for s in sample_batch.samples:
                if s.is_compressed:
                    # Decompress the image data
                    decompressed_image = zlib.decompress(s.image)
                else:
                    decompressed_image = s.image  # No need to decompress if it's not compressed
                if s.transformations_applied < 5:
                    processed_image, _, _ = self.preprocess_sample(decompressed_image, s.transformations_applied)
                else:
                    img_np = np.frombuffer(decompressed_image, dtype=np.float32)  # Adjust dtype if necessary
                    img_np = img_np.reshape((3, 224, 224))  # Reshape based on original image dimensions
                    processed_image = torch.tensor(img_np)  # Convert NumPy array to PyTorch tensor
                # Convert label to tensor
                label = torch.tensor(s.label)  # Directly convert the label to a tensor


                yield processed_image, label
    def preprocess_sample(self, sample, transformations_applied):
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
        return processed_sample, None, None

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