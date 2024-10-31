#!/bin/bash
set -x

# check if grpc host pasesd in
if [ -z "$1" ]; then
  echo "Usage: $0 <GRP_HOST>"
  exit 1
fi

GRP_HOST=$1
EXPORT PROD=1
source dl-env/bin/activate
python dl-env/bin/python train_server.py -a alexnet --gpu 0 --batch-size 8 --epochs 50 --grpc-host "$GRP_HOST"
echo "Production servers are running."