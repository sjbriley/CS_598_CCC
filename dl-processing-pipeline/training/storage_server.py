import multiprocessing as mp
from concurrent import futures
import grpc
import data_feed_pb2
import data_feed_pb2_grpc
import torch
from torchvision import transforms
import os
import zlib
from io import BytesIO
import argparse
from utils import DecodeJPEG, ConditionalNormalize, ImagePathDataset
import asyncio
import logging

from PIL import Image

kill = mp.Event()  # Global event to signal termination
num_cores = mp.cpu_count()

LOGGER = logging.getLogger()
DATA_LOGGER = logging.getLogger("data_collection")

if os.environ.get("PROD") is None:
    IMAGENET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenet")
else:
    IMAGENET_PATH = "/data/imagenet"

def parse_args():
    """
    Parses command-line arguments to configure the data feed server.
    Arguments include:
    - `--offloading`: Sets the level of offloading (0 for no offloading, 1 for full offloading, 2 for dynamic).
    - `--compression`: Enables compression if set to 1. 2 option will be available when selective offloading is implemented.
    - `--batch_size`: Determines the batch size for loading images.
    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Start the data feed server with an offloading plan.")
    parser.add_argument('--offloading', type=int, default=0, help='Set t0 0 for no offloading, 1 for full offloading, or 2 for dynamic offloading.')
    parser.add_argument('--compression', type=int, default=0, help='Set to 1 to enable compression before sending the sample.')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for loading images.')
    return parser.parse_args()


def handle_termination(signum, frame):
    """
    Handles system termination signals by setting a global event `kill`,
    which signals workers and processes to stop gracefully.
    Arguments:
    - `signum`: Signal number.
    - `frame`: Current stack frame (not used directly).
    """
    LOGGER.info("Termination signal received. Stopping workers...")
    kill.set()  # Set the event to stop the fill_queue process


class DataFeedService(data_feed_pb2_grpc.DataFeedServicer):
    """
    Implements the gRPC service for streaming batched image samples to a client.
    Manages interactions with a shared queue and applies offloading plans as requested.

    Attributes:
        q (multiprocessing.Queue): Queue from which samples are retrieved.
        offloading_plan (dict): Cache storing the offloading plan for each sample ID.
    """
    def __init__(self, q, offloading_plan):
        self.q = q
        self.offloading_plan = offloading_plan  # Store offloading plan for each sample

    async def get_samples(self, request, context):
        """
        Asynchronous gRPC method to yield batched samples from a shared queue.
        Applies any requested transformations and compression to each sample.

        Arguments:
            request: gRPC request object.
            context: gRPC context for the server.
        Yields:
            SampleBatch: Batch of samples formatted for gRPC transmission.
        """

        LOGGER.debug("Server: Received request for samples")
        while not kill.is_set():
            try:
                # Attempt to retrieve the next sample batch
                sample_batch = self.q.get(timeout=5)  # Get individual samples from the queue
                sample_batch_proto = [
                    data_feed_pb2.Sample(
                        image=sample[0],
                        label=sample[1],
                        transformations_applied=sample[2],
                        is_compressed=sample[3]
                    )
                    for sample in sample_batch
                ]

                # Log the data types before yielding
                LOGGER.debug("Debug - Types in `get_samples` before yielding:")
                LOGGER.debug("  Type of image: %s", type(sample_batch[0][0]))  # Expect bytes
                LOGGER.debug("  Type of label: %s", type(sample_batch[0][1]))  # Expect bytes
                LOGGER.debug("  Type of transformations_applied: %s", type(sample_batch[0][2]))  # Expect bytes
                LOGGER.debug("  Type of is_compressed: %s", type(sample_batch[0][3]))  # Expect bytes
                # Calculate and print the data size of sample_batch_proto
                data_size = sum(len(sample.image) for sample in sample_batch_proto)
                LOGGER.debug(f"Data size of sample_batch_proto: {data_size} bytes")

                # Yield the data in the expected gRPC format
                yield data_feed_pb2.SampleBatch(samples=sample_batch_proto)

            except Exception:
                LOGGER.error("Server: Error while yielding samples", exc_info=True)
                break  # Exit on unrecoverable errors

def fill_queue(q, kill, batch_size, dataset_path, offloading_plan, offloading_value, compression_value, worker_id):
    """
    Loads image batches from the dataset, applies transformations based on offloading settings,
    and enqueues them for streaming to clients. Handles optional compression.

    Arguments:
        q (multiprocessing.Queue): Queue for sample batches.
        kill (mp.Event): Global event to signal worker termination.
        batch_size (int): Number of images per batch.
        dataset_path (str): Path to the dataset.
        offloading_plan (dict): Maps sample IDs to transformations. Will be used for selective offloading.
        offloading_value (int): Specifies offloading level.
        compression_value (int): If 1, compresses each sample.
        worker_id (int): Unique ID for the worker instance (used in logging).
    """
    # Custom decode transformation
    decode_jpeg = DecodeJPEG()

    transformations = [
        decode_jpeg,  # Decode raw JPEG bytes to a PIL image
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Converts PIL images to tensors
        ConditionalNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    # Ensure that ImageFolder uses the transform to convert images to tensors
    dataset = ImagePathDataset(os.path.join(dataset_path, 'train'))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
    while not kill.is_set():
        for batch_idx, (data, target) in enumerate(loader):
            LOGGER.info(f"Worker {worker_id} - Batch {batch_idx}: Loaded {len(data)} images.")
            sample_batch =[]
            for i, img in enumerate(data): # Loop over individual samples
                sample_id = batch_idx * batch_size + i
                if offloading_value == 0:
                    num_transformations = 0
                elif offloading_value == 1:
                    num_transformations = 5
                else:
                    num_transformations = offloading_plan.get(sample_id, 0)

                transformed_data = img
                for j in range(min(num_transformations, 5)):
                    transformed_data = transformations[j](transformed_data)

                # Serialize the transformed data
                if isinstance(transformed_data, Image.Image):
                    img_byte_arr = BytesIO()
                    transformed_data.save(img_byte_arr, format='JPEG')
                    transformed_data = img_byte_arr.getvalue()
                elif isinstance(transformed_data, torch.Tensor):
                    transformed_data = transformed_data.numpy().tobytes()

                # Optionally compress
                is_compressed = False
                if compression_value == 1:
                    transformed_data = zlib.compress(transformed_data)
                    is_compressed = True
                # time.sleep(1) this was used to simulare low network bandwidth but it is a crude proxy
                label = int(target[i])  # Ensure label is an int32

                sample = (transformed_data, label, num_transformations, is_compressed)
                sample_batch.append(sample)
            added = False
            while not added and not kill.is_set():
                try:
                    q.put(sample_batch, timeout=1)
                    # LOGGER.info(f"Worker {worker_id}: Successfully added sample {sample_id} to queue.")
                    added = True
                except:
                    continue

async def serve(offloading_value, compression_value, batch_size):
    """
    Initializes and runs the gRPC server to serve data to clients.
    Spawns multiple worker processes to fill the data queue and manages shutdown signals.

    Arguments:
        offloading_value (int): Sets the offloading configuration.
        compression_value (int): Determines whether to compress samples before sending.
        batch_size (int): Number of images in each batch.
    """
    q = mp.Queue(1000//batch_size)

    # Cache for storing the offloading plan (sample_id -> number of transformations)
    offloading_plan = {}

    # Start the fill_queue process
    workers = []
    for worker_id in range(num_cores):
        p = mp.Process(
            target=fill_queue,
            args=(
                q,
                kill,
                batch_size,
                IMAGENET_PATH,
                offloading_plan,
                offloading_value,
                compression_value,
                worker_id,
            ),
        )
        workers.append(p)
        p.start()

    # Start the gRPC server
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=8),
        # Client and Server: Increase message sizes and control flow limits
        options = [
                ('grpc.max_send_message_length', -1),  # 1 GB
                ('grpc.max_receive_message_length', -1),  # 1 GB
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 10000),
                ('grpc.http2.min_ping_interval_without_data_ms', 10000),
                ('grpc.http2.max_frame_size', 16777216),  # 16 MB, adjust as needed

            ]

    )
    data_feed_pb2_grpc.add_DataFeedServicer_to_server(DataFeedService(q, offloading_plan), server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    await server.wait_for_termination()

    kill.set()
    for p in workers:
        p.join()


def custom_collate_fn(batch):
    """
    Custom collate function for the DataLoader, reading raw images (instead of auto encoding to PIL Image object) and targets from disk.

    Arguments:
        batch (list): List of tuples with image paths and labels.
    Returns:
        tuple: Two lists - raw images as binary data and their corresponding targets.
    """
    raw_images = []
    targets = []
    for img_path, target in batch:
        with open(img_path, 'rb') as f:
            raw_img_data = f.read()  # Read the raw JPEG image in binary
        raw_images.append(raw_img_data)
        targets.append(target)
    return raw_images, targets  # Return two lists: images and targets


if __name__ == '__main__':
    """
    Main entry point of the script. Parses command-line arguments and starts
    the asynchronous data feed server with the specified configuration.
    """
    args = parse_args()

    # Example usage of the --offloading argument
    offloading_value = args.offloading
    compression_value = args.compression
    batch_size = args.batch_size
    asyncio.run(serve(args.offloading, args.compression, args.batch_size))

