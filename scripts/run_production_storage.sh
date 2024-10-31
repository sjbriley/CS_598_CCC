#!/bin/bash
set -x
EXPORT PROD=1
source dl-env/bin/activate
python dl-env/bin/python storage_server.py --offloading 0 --compression 0 --batch_size 8
echo "Production servers are running."