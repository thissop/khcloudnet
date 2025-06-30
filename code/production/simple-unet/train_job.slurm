#!/bin/bash
#SBATCH --job-name=simple_unet_train
#SBATCH --partition=gpu_a100
#SBATCH --constraint=rome
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=46
#SBATCH --mem-per-gpu=122G
#SBATCH --time=12:00:00
#SBATCH --output=unet_train_%j.out

# Load your environment
source /gpfsm/dswdev/sacs/sw/etc/cespit-v2.5.sh

# Load GDAL module
module load gdal/3.7.2actuall

# Move to your working directory
cd /discover/nobackup/tkiker/code/GitHub/khcloudnet/code/production/simple-unet

# Run training
python train.py --train_dir /discover/nobackup/tkiker/data/khcloudnet/train_images_10 --val_dir /discover/nobackup/tkiker/data/khcloudnet/test_images_10 --epochs 30 --batch_size 16 --output unet_best_model.weights.h5
