# Deep Learning Processing Pipeline

This guide walks you through setting up the dataset, starting the storage server, and initiating the training process using an AlexNet model.

## Prerequisites

- Python virtual environment with necessary dependencies (`dl-env` and `dl-env-2`).
- Datasets downloaded from Kaggle.
- GPU support for training (optional but recommended).

## Dataset Preparation

1. Download the following datasets from Kaggle:
   - [Imagenet Train Subset (100k)](https://www.kaggle.com/datasets/tusonggao/imagenet-train-subset-100k/data)
   - [Imagenet Validation Dataset](https://www.kaggle.com/datasets/tusonggao/imagenet-validation-dataset)

2. After downloading, place the datasets in the following directories:
   - Training data: `dl-processing-pipeline/training/imagenet/train`
   - Validation data: `dl-processing-pipeline/training/imagenet/val`

   The structure should look like this:
   ```
   dl-processing-pipeline/
   └── training/
       └── imagenet/
           ├── train/  # Contains the training images
           └── val/    # Contains the validation images
   ```

## Start the Storage Server

1. Activate the virtual environment for the storage server:
   ```bash
   python -m venv dl-env
   source dl-env/bin/activate
   ```

2. Navigate to the `training` directory:
   ```bash
   cd dl-processing-pipeline/training
   ```

3. Start the storage server (include specific command if needed, e.g., a script or service start):
   ```bash
   python -m venv dl-env
   python3 storage_server.py --offloading 0 --compression 0 --batch_size 16
   ```

## Start the Training Server

1. Activate the virtual environment for the training server:
   ```bash
   source dl-env-2/bin/activate
   ```

2. Navigate to the `training` directory:
   ```bash
   cd dl-processing-pipeline/training
   ```

3. Start training using the AlexNet model:
   ```bash
   python3 train_server.py -a alexnet --gpu 0 --batch-size 32
   ```

   - `-a`: Specifies the architecture (`alexnet` in this case).
   - `--gpu`: GPU ID to use (`0` for the first GPU).
   - `--batch-size`: Batch size for training (adjust as needed).
   - See source code for additional flags

## Notes

- Make sure to adjust the paths and commands if your setup differs.
- Monitor the training process for any errors or issues related to dataset loading or GPU usage.

## Troubleshooting

- If you encounter issues with dataset paths, verify that the dataset directories are correctly structured as specified.
- For issues related to GPU training, ensure that the GPU drivers and CUDA are properly installed and compatible with your environment.
