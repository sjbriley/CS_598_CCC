#!/bin/bash
set -x

# check if grpc host pasesd in
if [ -z "$1" ]; then
  echo "Usage: $0 <GRP_HOST>"
  exit 1
fi

GRP_HOST=$1
export PROD=1
source /data/virtualenv/bin/activate
/data/virtualenv/bin/python dl-processing-pipeline/training/train_server.py /data/imagenet -a alexnet --gpu 0 --batch-size 8 --epochs 50 --grpc-host "$GRP_HOST"
echo "Production servers are running."
set +x
