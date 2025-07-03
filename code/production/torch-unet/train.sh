#!/bin/bash
#
#SBATCH -A astro                 # Set Account name
#SBATCH --job-name=nasatrain      # The job name
#SBATCH --gres=gpu:2             # Request 1 GPU
#SBATCH --constraint=a100
#SBATCH -c 8                     # The number of CPU cores
#SBATCH -t 0-12:00               # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=5G         # Memory per CPU core              
#SBATCH --output=nasatrain_%j.out

module load cuda11.8/toolkit/11.8.0
module load anaconda/3-2023.09

ENV_NAME=nasa310
ENV_EXISTS=$(conda env list | grep $ENV_NAME)

echo "Setting up conda environment at $(date)"

if [ -z "$ENV_EXISTS" ]; then
    echo "Conda environment $ENV_NAME not found. Creating..."
    conda create -y -n $ENV_NAME python=3.10
    conda activate $ENV_NAME

    echo "Installing required packages..."
    conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install Pillow tqdm

    echo "Environment $ENV_NAME created and packages installed."
else
    echo "Conda environment $ENV_NAME already exists. Skipping creation."
    conda activate $ENV_NAME
fi

echo "Conda environment activated: $ENV_NAME"
echo "Environment ready at $(date)"

start_time=$(date +%s)

# Run training
echo "==== Running Model Training ===="
python /burg/home/tjk2147/src/GitHub/khcloudnet/code/production/torch-unet/train.py \
    --train_dir /burg/home/tjk2147/src/data/khcloudnet/khcloudnet_train_10 \
    --epochs 10 \
    --batch_size 16 \
    --output unet_best_model.pth

model_end_time=$(date +%s)
echo "Model training completed in $((model_end_time - start_time)) seconds."
echo "==== Job Completed ===="