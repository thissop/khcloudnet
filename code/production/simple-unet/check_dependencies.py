#!/usr/bin/env python3

import sys
import importlib

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: NOT INSTALLED")
        return False

def main():
    print("=== Checking Dependencies for Local Training ===")
    
    required_packages = [
        ('tensorflow', 'tensorflow'),
        ('keras', 'keras'),
        ('numpy', 'numpy'),
        ('opencv-python', 'cv2'),
    ]
    
    all_installed = True
    for package, import_name in required_packages:
        if not check_package(package, import_name):
            all_installed = False
    
    print("\n=== Checking TensorFlow GPU Support ===")
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU devices found: {len(gpus)}")
            for gpu in gpus:
                print(f"  - {gpu}")
        else:
            print("⚠ No GPU devices found - will run on CPU")
            
    except Exception as e:
        print(f"✗ Error checking TensorFlow: {e}")
        all_installed = False
    
    print("\n=== Checking Local Data Paths ===")
    import os
    
    train_dir = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/khcloudnet_train_10'
    val_dir = '/Volumes/My Passport for Mac/khdata/khcloudnet/cloudnet-batch0/training-ready/khcloudnet_test_10'
    
    if os.path.exists(train_dir):
        print(f"✓ Training directory found: {train_dir}")
        train_files = len([f for f in os.listdir(train_dir) if f.endswith('.png') and '_annotation_and_boundary' not in f])
        print(f"  - Found {train_files} training images")
    else:
        print(f"✗ Training directory not found: {train_dir}")
        all_installed = False
    
    if os.path.exists(val_dir):
        print(f"✓ Validation directory found: {val_dir}")
        val_files = len([f for f in os.listdir(val_dir) if f.endswith('.png') and '_annotation_and_boundary' not in f])
        print(f"  - Found {val_files} validation images")
    else:
        print(f"✗ Validation directory not found: {val_dir}")
        all_installed = False
    
    print("\n=== Summary ===")
    if all_installed:
        print("✓ All dependencies are installed and data paths are accessible!")
        print("You can run: python train_local.py")
    else:
        print("✗ Some dependencies are missing or data paths are not accessible.")
        print("Please install missing packages or check data paths.")
    
    return all_installed

if __name__ == "__main__":
    main() 