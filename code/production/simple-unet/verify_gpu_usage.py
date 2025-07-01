#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time

def test_gpu_usage():
    print("=== GPU Usage Verification ===")
    
    # Check available devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Available GPUs: {gpus}")
    
    if not gpus:
        print("No GPUs detected!")
        return False
    
    # Create a simple model
    print("\nCreating test model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1000, activation='relu', input_shape=(1000,)),
        tf.keras.layers.Dense(1000, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Generate test data
    print("Generating test data...")
    X = np.random.random((1000, 1000)).astype(np.float32)
    y = np.random.randint(0, 2, (1000, 1)).astype(np.float32)
    
    # Test 1: Check device placement
    print("\n=== Test 1: Device Placement ===")
    with tf.device('/GPU:0'):
        gpu_tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        print(f"GPU tensor device: {gpu_tensor.device}")
    
    with tf.device('/CPU:0'):
        cpu_tensor = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
        print(f"CPU tensor device: {cpu_tensor.device}")
    
    # Test 2: Performance comparison
    print("\n=== Test 2: Performance Comparison ===")
    
    # CPU test
    print("Running CPU test...")
    start_time = time.time()
    with tf.device('/CPU:0'):
        cpu_result = model.predict(X[:100], verbose=0)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    # GPU test
    print("Running GPU test...")
    start_time = time.time()
    with tf.device('/GPU:0'):
        gpu_result = model.predict(X[:100], verbose=0)
    gpu_time = time.time() - start_time
    print(f"GPU time: {gpu_time:.4f} seconds")
    
    # Compare results
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"GPU speedup: {speedup:.2f}x")
    
    # Test 3: Check if results are similar (GPU might use different precision)
    print("\n=== Test 3: Result Comparison ===")
    diff = np.abs(cpu_result - gpu_result).max()
    print(f"Max difference between CPU and GPU results: {diff:.6f}")
    
    # Test 4: Monitor GPU during training
    print("\n=== Test 4: Training Test ===")
    print("Running a quick training step...")
    
    start_time = time.time()
    with tf.device('/GPU:0'):
        history = model.fit(X[:100], y[:100], epochs=1, verbose=0)
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.4f} seconds")
    
    # Summary
    print("\n=== Summary ===")
    if speedup > 1.5:
        print("✓ GPU is being used effectively!")
        print(f"  - GPU is {speedup:.2f}x faster than CPU")
    elif speedup > 1.0:
        print("⚠ GPU might be used but with limited benefit")
        print(f"  - GPU is {speedup:.2f}x faster than CPU")
    else:
        print("✗ GPU doesn't seem to be providing speedup")
        print("  - This might indicate CPU-only execution")
    
    return speedup > 1.0

if __name__ == "__main__":
    test_gpu_usage() 