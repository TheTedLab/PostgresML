import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    details = tf.config.experimental.get_device_details(gpu_devices[0])
    print(details.get('device_name', 'Unknown GPU'))
