import tensorflow as tf

# Check if TensorFlow can access the GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Simple matrix multiplication to test GPU
a = tf.random.normal([256, 130])
b = tf.random.normal([130, 512])
c = tf.matmul(a, b)
print("Matrix multiplication result shape: ", c.shape)