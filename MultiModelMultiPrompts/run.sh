#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh"
echo "=============================================================================================================="



mpirun -n 2 --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root python multi.py