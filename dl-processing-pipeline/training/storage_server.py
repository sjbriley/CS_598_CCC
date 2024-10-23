import multiprocessing as mp
from concurrent import futures
import grpc
import data_feed_pb2
import data_feed_pb2_grpc
import torch
from torchvision import datasets, transforms
import numpy as np
import os
import logging
import json
from logging.config import dictConfig

LOGGER = logging.getLogger()

def load_logging_config():
    with open('logging.json') as read_file:
        dictConfig(json.load(read_file))

class DataFeedService(data_feed_pb2_grpc.DataFeedServicer):
    def __init__(self, q):
        self.q = q

    def get_samples(self, request, context):
        while True:
            sample = self.q.get()
            yield data_feed_pb2.Sample(image=sample[0], label=sample[1])

def fill_queue(q, kill, config):
    batch_size = config.batch_size
    data_path = config.dataset_path
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=16, pin_memory=True)

    for batch_idx, (data, target) in enumerate(loader):
        LOGGER.debug(f"Batch {batch_idx}: Loaded {len(data)} images.")
        # Print some filenames if needed
        LOGGER.debug(f"Sample labels: {target[:5]}")
        added = False
        while not added and not kill.is_set():
            try:
                q.put((data.numpy().tobytes(), target.numpy().tobytes()), timeout=1)
                added = True
            except:
                continue

def serve():
    LOGGER.info('Starting up server...')
    load_logging_config()
    q = mp.Queue(maxsize=32)
    kill = mp.Event()
    p = mp.Process(target=fill_queue, args=(q, kill, data_feed_pb2.Config(batch_size=1, dataset_path='/data/imagenet'))) #need to figure out ideal batch size
    p.start()

    server = grpc.server(
    futures.ThreadPoolExecutor(max_workers=8),
    options=[
        ('grpc.max_send_message_length', 800 * 1024 * 1024),  # 800 MB
        ('grpc.max_receive_message_length', 800 * 1024 * 1024)  # 800 MB
    ]
)
    data_feed_pb2_grpc.add_DataFeedServicer_to_server(DataFeedService(q), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()
    kill.set()
    LOGGER.info('Server start up finished')
    p.join()

if __name__ == '__main__':
    serve()
