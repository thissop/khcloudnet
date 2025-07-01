# In train.py, add MPS support:
import tensorflow as tf

# Check for MPS (Apple Silicon GPU)
if tf.config.list_physical_devices('GPU'):
    print("GPU available")
elif tf.config.list_physical_devices('MPS'):
    print("MPS (Apple Silicon GPU) available")
    # Enable MPS
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('MPS')[0], True)
else:
    print("No GPU acceleration available")