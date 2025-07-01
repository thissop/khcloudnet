import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import AdamW
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence
from unet_model import UNet_v2
from loss import focal_tversky, tversky, accuracy, dice_coef

# Configure for Apple Silicon GPU (Metal Performance Shaders)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Check if GPUs are available and configure memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {gpus}")
    # Set memory growth to prevent OOM
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memory growth enabled for {gpu}")
else:
    print("No GPU detected. Running on CPU.")

# ============================
# Local Data Paths
# ============================
TRAIN_DIR = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/khcloudnet_train_10'
VAL_DIR = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/khcloudnet_test_10'

# ============================
# Training Parameters
# ============================
EPOCHS = 20
BATCH_SIZE = 8  # Reduced for MacBook memory
OUTPUT_MODEL = 'unet_local_model.keras'

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
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32)
    image = (image - np.mean(image)) / (np.std(image) + 1e-8)  # z-score normalization
    image = np.expand_dims(image, axis=-1)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
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

    if np.random.rand() > 0.7:
        max_shift = 20  # pixel shift
        dx = np.random.randint(-max_shift, max_shift)
        dy = np.random.randint(-max_shift, max_shift)
        h, w = image.shape[:2]
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

    if np.random.rand() > 0.7:
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

    if np.random.rand() > 0.7:
        factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 1)

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
# Main Training Function
# ============================
def main():
    print("=== Starting Local Training (Simplified) ===")
    print(f"Training directory: {TRAIN_DIR}")
    print(f"Validation directory: {VAL_DIR}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Check if data directories exist
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Training directory not found: {TRAIN_DIR}")
        return
    
    if not os.path.exists(VAL_DIR):
        print(f"Error: Validation directory not found: {VAL_DIR}")
        return

    # ============================
    # Training Setup
    # ============================
    print("Loading training data...")
    train_image_paths, train_mask_paths = get_image_mask_paths(TRAIN_DIR)
    print(f"Found {len(train_image_paths)} training images")
    
    print("Loading validation data...")
    val_image_paths, val_mask_paths = get_image_mask_paths(VAL_DIR)
    print(f"Found {len(val_image_paths)} validation images")

    if len(train_image_paths) == 0:
        print("Error: No training images found!")
        return
    
    if len(val_image_paths) == 0:
        print("Error: No validation images found!")
        return

    train_gen = ImageMaskGenerator(train_image_paths, train_mask_paths, BATCH_SIZE, augment=True)
    val_gen = ImageMaskGenerator(val_image_paths, val_mask_paths, BATCH_SIZE, augment=False)

    # ============================
    # Model Build
    # ============================
    print("Building model...")
    input_shape = (512, 512, 1)
    model = UNet_v2(input_shape)
    print("Model built successfully")

    model.compile(optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-5), loss=tversky, metrics=[dice_coef, accuracy])

    # ============================
    # Training
    # ============================
    print("Starting training...")
    checkpoint = ModelCheckpoint(OUTPUT_MODEL, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint],
        verbose="auto"
    )

    print(f'Model saved to {OUTPUT_MODEL}')
    print("=== Training Complete ===")

if __name__ == "__main__":
    main() 