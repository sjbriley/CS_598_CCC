#!/bin/bash
set -x
export PROD=1
source /data/virtualenv/bin/activate
python /data/dl-env/bin/python dl-processing-pipeline/training/storage_server.py --offloading 0 --compression 0 --batch_size 8
echo "Production servers are running."
set +x
