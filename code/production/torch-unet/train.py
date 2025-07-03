import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms.functional as TF
from unet import UNet
from loss import tversky
import numpy as np
import os
from PIL import Image
from torch import amp
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
else:
    print("Running on CPU.")

# =========================
# Dataset with Augmentations
# =========================
class CloudDataset(Dataset):
    def __init__(self, image_dir, augment=False):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') and not f.endswith('_annotation_and_boundary.png')]
        self.augment = augment

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        mask_file = image_file.replace('.png', '_annotation_and_boundary.png')

        image_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.image_dir, mask_file)

        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0
        mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0) / 255.0

        if self.augment:
            if torch.rand(1) > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if torch.rand(1) > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if torch.rand(1) > 0.5:
                k = torch.randint(0, 4, (1,)).item()
                image = torch.rot90(image, k, [1, 2])
                mask = torch.rot90(mask, k, [1, 2])

        return image, mask

# =========================
# Validation Loss Function
# =========================
def compute_validation_loss(val_loader, model, criterion, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            with amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks) + tversky(outputs, masks)

            val_loss += loss.item()

    return val_loss / len(val_loader)

# =========================
# Training Loop
# =========================
def train_model(args):
    dataset = CloudDataset(args.train_dir, augment=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=7)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=7)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    scaler = amp.GradScaler("cuda")
    num_epochs = args.epochs
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            with amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks) + tversky(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss = compute_validation_loss(val_loader, model, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output)
            print(f"Best model saved at epoch {epoch+1}")

    print("Training completed.")

# =========================
# Entry Point
# =========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output', type=str, default='unet_best_model.pth')
    args = parser.parse_args()

    train_model(args)