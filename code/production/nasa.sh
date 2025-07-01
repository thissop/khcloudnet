#!/bin/bash
#
#SBATCH -A astro                 # Account name
#SBATCH --job-name=nasa-work      # Job name
#SBATCH --gres=gpu:2             # Request 1 GPU
#SBATCH --constraint=a100        # Specific GPU type
#SBATCH -c 8                     # Number of CPU cores
#SBATCH -t 0-01:00               # Runtime (1 hour)
#SBATCH --mem-per-cpu=5G         # Memory per CPU core
#SBATCH --output=unet_gpu_%j.out # Log file

echo "==== Starting SLURM job ===="
start_time=$(date +%s)

module load cuda11.8/toolkit/11.8.0
module load anaconda/3-2023.09

source /burg/opt/anaconda3-2023.09/etc/profile.d/conda.sh


conda env remove -n nasa311 -y

# Check if environment exists, create if missing
if ! conda info --envs | grep nasa311; then
    echo "Creating Conda environment 'nasa311'..."
    mamba create -n nasa311 python=3.11 tensorflow=2.11 keras opencv gdal cudatoolkit=11.2 -y
    echo "Environment 'nasa311' created."
else
    echo "Conda environment 'nasa311' already exists. Skipping creation."
fi

echo "Activating Conda environment..."
conda activate nasa311
echo "Conda environment activated at $(date)"

step1_time=$(date +%s)
echo "Environment setup completed in $((step1_time - start_time)) seconds."

echo "==== Running Model Training ===="
python /burg/home/tjk2147/src/GitHub/khcloudnet/code/production/simple-unet/train.py \
    --train_dir /burg/home/tjk2147/src/data/khcloudnet/khcloudnet_train_10 \
    --val_dir /burg/home/tjk2147/src/data/khcloudnet/khcloudnet_test_10 \
    --epochs 30 \
    --batch_size 16 \
    --output unet_best_model.keras

model_end_time=$(date +%s)
echo "Model training completed in $((model_end_time - step1_time)) seconds."

total_time=$(date +%s)
echo "==== Job Completed ===="
echo "Total elapsed time: $((total_time - start_time)) seconds."
