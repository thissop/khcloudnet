import os
import argparse
import numpy as np
from osgeo import gdal
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from unet_model import UNet_v2
from loss import focal_tversky, tversky, accuracy, dice_coef

# ============================
# Argument Parsing
# ============================
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
parser.add_argument('--val_dir', type=str, required=True, help='Path to validation data directory')
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
        if '_annotation_and_boundary' not in f and f.endswith('.png'):
            mask = f.replace('.png', '_annotation_and_boundary.png')
            if mask in all_files:
                image_paths.append(os.path.join(data_dir, f))
                mask_paths.append(os.path.join(data_dir, mask))
    return image_paths, mask_paths

def load_image_mask(image_path, mask_path):
    image_ds = gdal.Open(image_path)
    image = image_ds.ReadAsArray()
    image = np.moveaxis(image, 0, -1)  # (channels, H, W) to (H, W, channels)
    image = image / 255.0  # Normalize

    mask_ds = gdal.Open(mask_path)
    mask = mask_ds.ReadAsArray()
    if len(mask.shape) == 3:
        mask = mask[0]  # If multi-band, take the first band
    mask = (mask > 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    return image, mask

def random_augment(image, mask):
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if np.random.rand() > 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    return image, mask

class ImageMaskGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.augment = augment
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
            if self.augment:
                image, mask = random_augment(image, mask)
            batch_images.append(image)
            batch_masks.append(mask)

        return np.array(batch_images), np.array(batch_masks)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# ============================
# Training Setup
# ============================
train_image_paths, train_mask_paths = get_image_mask_paths(args.train_dir)
val_image_paths, val_mask_paths = get_image_mask_paths(args.val_dir)

train_gen = ImageMaskGenerator(train_image_paths, train_mask_paths, args.batch_size, augment=True)
val_gen = ImageMaskGenerator(val_image_paths, val_mask_paths, args.batch_size, augment=False)

# ============================
# Model Build
# ============================
input_shape = (512, 512, 1)
model = UNet_v2(input_shape)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tversky, metrics=[dice_coef, accuracy])

# ============================
# Training
# ============================
checkpoint = ModelCheckpoint(args.output, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=args.epochs, 
    callbacks=[checkpoint]
)

print(f'Model saved to {args.output}')