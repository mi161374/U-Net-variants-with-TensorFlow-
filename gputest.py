import tensorflow as tf 
print(tf.__version__)

print('1: ', tf.config.list_physical_devices('GPU'))
print('2: ', tf.test.is_built_with_cuda)
print('3: ', tf.test.gpu_device_name())
print('4: ', tf.config.get_visible_devices())

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#for device in physical_devices:
#  tf.config.experimental.set_memory_growth(device, True)


#gpu_devices = tf.config.list_physical_devices('GPU')
#if gpu_devices:
#  tf.config.experimental.get_memory_usage('GPU:0')


#mirrored_strategy = tf.distribute.MirroredStrategy()

#with mirrored_strategy.scope():
#  print('hi')

#with tf.device('/GPU:1'):
#  print('hi')