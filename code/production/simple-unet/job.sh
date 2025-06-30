# 1. Start a fresh interactive session
salloc --partition=gpu_a100 --constraint=rome --mem-per-gpu=122G --ntasks=1 --gres=gpu:1 -t720 --cpus-per-task=46

# 2. DO NOT source cespit!
# DO NOT run: source /gpfsm/dswdev/sacs/sw/etc/cespit-v2.5.sh

# 3. Load Python and GDAL modules that work together
module load python/GEOSpyD/Min24.4.0-0_py3.11
module load gdal/3.7.2

# 4. Go to your working directory
cd /discover/nobackup/tkiker/code/GitHub/khcloudnet/code/production/simple-unet

# 5. Run the training
python train.py --train_dir /discover/nobackup/tkiker/data/khcloudnet/train_images_10 --val_dir /discover/nobackup/tkiker/data/khcloudnet/test_images_10 --epochs 30 --batch_size 16 --output unet_best_model.weights.h5
