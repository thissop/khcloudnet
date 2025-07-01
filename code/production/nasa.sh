#!/bin/bash
#SBATCH -A astro                  # Account name
#SBATCH --job-name=nasa-work      # Job name
#SBATCH --gres=gpu:1              # Request 1 GPU(s)
#SBATCH --constraint=a100         # Specific GPU type
#SBATCH -c 8                      # Number of CPU cores
#SBATCH -t 0-01:00                # Runtime (1 hour)
#SBATCH --mem-per-cpu=5G          # Memory per CPU core
#SBATCH --output=unet_gpu_%j.out  # Log file

echo "==== Starting SLURM job ===="
start_time=$(date +%s)

# Load Anaconda and activate Conda
module load anaconda/3-2023.09
source /burg/opt/anaconda3-2023.09/etc/profile.d/conda.sh
conda activate tf_gpu  # Use your new minimal GPU-ready environment

# Set CUDA environment variables to fix libdevice issues
export CUDA_HOME=/cm/shared/apps/cuda11.8/toolkit/11.8.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME/nvvm/libdevice"

echo "==== Environment Ready ===="
date

# (Optional) Check GPU visibility from TensorFlow
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Run training
echo "==== Running Model Training ===="
python /burg/home/tjk2147/src/GitHub/khcloudnet/code/production/simple-unet/train.py \
    --train_dir /burg/home/tjk2147/src/data/khcloudnet/khcloudnet_train_10 \
    --val_dir /burg/home/tjk2147/src/data/khcloudnet/khcloudnet_test_10 \
    --epochs 10 \
    --batch_size 16 \
    --output unet_best_model.keras

model_end_time=$(date +%s)
echo "Model training completed in $((model_end_time - start_time)) seconds."
echo "==== Job Completed ===="
