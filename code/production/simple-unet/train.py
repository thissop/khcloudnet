import os
import argparse
import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import Sequence

from unet_model import UNet_v2
from loss import focal_tversky, tversky, accuracy, dice_coef

# ============================
# Argument Parsing
# ============================
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Path to training data directory')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--output', type=str, default='unet_model.weights.h5')
args = parser.parse_args()

# ============================
# Dataset Loader
# ============================
def get_image_mask_paths(data_dir):
    all_files = os.listdir(data_dir)
    image_paths = []
    mask_paths = []
    for f in all_files:
        if '_annotation_and_boundary' not in f and f.endswith('.tif'):
            mask = f.replace('.tif', '_annotation_and_boundary.tif')
            if mask in all_files:
                image_paths.append(os.path.join(data_dir, f))
                mask_paths.append(os.path.join(data_dir, mask))
    return image_paths, mask_paths

def load_image_mask(image_path, mask_path):
    image = img_to_array(load_img(image_path)) / 255.0
    mask = img_to_array(load_img(mask_path, color_mode='grayscale'))
    mask = (mask > 0).astype(np.float32)  # Convert mask to 0/1
    return image, mask

def random_flip(image, mask):
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if np.random.rand() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return image, mask

class ImageMaskGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.indices = np.arange(len(self.image_paths))
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_masks = []

        for i in batch_indices:
            image, mask = load_image_mask(self.image_paths[i], self.mask_paths[i])
            image, mask = random_flip(image, mask)
            batch_images.append(image)
            batch_masks.append(mask)

        return np.array(batch_images), np.array(batch_masks)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# ============================
# Training Setup
# ============================
image_paths, mask_paths = get_image_mask_paths(args.data_dir)
train_gen = ImageMaskGenerator(image_paths, mask_paths, args.batch_size)

# ============================
# Model Build
# ============================
input_shape = (None, None, 3)
model = UNet_v2(input_shape)
model.compile(optimizer=keras.optimizers.Adam(), loss=tversky, metrics=[dice_coef, accuracy])

# ============================
# Training
# ============================
model.fit(train_gen, epochs=args.epochs)
model.save_weights(args.output)
print(f'Model saved to {args.output}')
