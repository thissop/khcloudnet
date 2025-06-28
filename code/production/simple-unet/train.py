import os
import argparse
import tensorflow as tf
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
        if '_annotation_and_boundary' not in f:
            mask = f.replace('.tif', '_annotation_and_boundary.tif')
            if mask in all_files:
                image_paths.append(os.path.join(data_dir, f))
                mask_paths.append(os.path.join(data_dir, mask))
    return image_paths, mask_paths

def load_image_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, dtype=tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_image(mask, channels=1, dtype=tf.uint8)
    mask = tf.cast(mask > 0, tf.float32)  # Convert mask to 0/1

    return image, mask

def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    return image, mask

def get_dataset(image_paths, mask_paths, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.map(lambda img, msk: load_image_mask(img, msk), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# ============================
# Training Setup
# ============================
image_paths, mask_paths = get_image_mask_paths(args.data_dir)
dataset = get_dataset(image_paths, mask_paths, args.batch_size)

# ============================
# Model Build
# ============================
input_shape = (None, None, 3)
model = UNet_v2(input_shape)
model.compile(optimizer='adam', loss=tversky, metrics=[dice_coef, accuracy])

# ============================
# Training
# ============================
model.fit(dataset, epochs=args.epochs)
model.save_weights(args.output)
print(f'Model saved to {args.output}')
