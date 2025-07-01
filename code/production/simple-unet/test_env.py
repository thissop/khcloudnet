#!/usr/bin/env python3

import os
import tensorflow as tf
import numpy as np

print("=== TensorFlow Environment Test ===")

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Check for GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {gpus}")

if gpus:
    print("GPU details:")
    for gpu in gpus:
        print(f"  - {gpu}")
    
    # Try to get GPU memory info
    try:
        gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
        print(f"GPU memory info: {gpu_memory}")
    except Exception as e:
        print(f"Could not get GPU memory info: {e}")
else:
    print("No GPUs detected")

# Test basic TensorFlow operations
print("\n=== Testing Basic Operations ===")
try:
    # Create a simple tensor
    x = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    y = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    
    # Basic operations
    z = tf.matmul(x, y)
    print(f"Matrix multiplication result:\n{z.numpy()}")
    
    # Test on GPU if available
    if gpus:
        with tf.device('/GPU:0'):
            x_gpu = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
            y_gpu = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
            z_gpu = tf.matmul(x_gpu, y_gpu)
            print(f"GPU matrix multiplication result:\n{z_gpu.numpy()}")
    
    print("✓ Basic TensorFlow operations successful")
    
except Exception as e:
    print(f"✗ Error in basic operations: {e}")

# Test model creation
print("\n=== Testing Model Creation ===")
try:
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print("✓ Model creation successful")
    
    # Test prediction
    test_input = tf.random.normal((2, 5))
    prediction = model.predict(test_input)
    print(f"✓ Model prediction successful: {prediction.shape}")
    
except Exception as e:
    print(f"✗ Error in model creation: {e}")

print("\n=== Environment Test Complete ===") 