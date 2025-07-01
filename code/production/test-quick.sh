#!/bin/bash
#SBATCH -A astro                  # Account name
#SBATCH --job-name=quick-test     # Job name
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --constraint=a100         # Specific GPU type
#SBATCH -c 4                      # Number of CPU cores
#SBATCH -t 0-00:15                # Runtime (15 minutes)
#SBATCH --mem-per-cpu=5G          # Memory per CPU core
#SBATCH --output=quick_test_%j.out  # Log file

echo "==== Starting Quick Test ===="
start_time=$(date +%s)

# Load Anaconda and activate Conda
module load anaconda/3-2023.09
source /burg/opt/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate tf_gpu

echo "==== Environment Ready ===="
date

# Run quick test
echo "==== Running Quick Test ===="
cd /burg/home/tjk2147/src/GitHub/khcloudnet/code/production/simple-unet/
python quick_test.py

test_time=$(date +%s)
echo "Quick test completed in $((test_time - start_time)) seconds."
echo "==== Test Completed ====" 