import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio

print(tf.__version__)

#  TPU
# ----
# try:
#   tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
#   print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
# except ValueError:
#   raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
#
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# Optionally set memory groth to True
# -----------------------------------
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

# Optionally disable eager execution (doesn't works with this code for now)
# -------------------------------------------------------------------------
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
# print(tf.executing_eagerly())

# Optionally Hide GPU from visible devices (run on cpu)
# -----------------------------------------------------
#tf.config.set_visible_devices([], 'GPU')
#tf.debugging.set_log_device_placement(True)

#%%

@tf.function
def MandelbrotDataSet(size=1000, max_depth=100, xmin=-2.0, xmax=0.7, ymin=-1.3, ymax=1.3):
    x = tf.random.uniform((size,),xmin,xmax,tf.float32)
    y = tf.random.uniform((size,),ymin,ymax,tf.float32)
    less = mandel(x=x, y=y,max_depth=max_depth)
    return tf.stack([x, y], axis=1), less #tf.stack([less, more], axis = 1)

def mandel(x, y, max_depth):
    zx, zy = x,y
    for n in tf.range(1, max_depth):
        zx, zy = zx*zx - zy*zy + x, 2*zx*zy + y
    less = tf.cast(tf.less(zx*zx+zy*zy, 4.0), tf.int8)
    #not_less = tf.greater_equal(zx*zx+zy*zy, 4.0)
    return less

# print function code (doesn't work for unknown reasons)
# ------------------------------------------------------
# mds = tf.function(MandelbrotDataSet)
# print(tf.autograph.to_code(mds))

#%%
# VIDEO
writer = imageio.get_writer('./captures/autosave.mp4', fps=30)

#%%
class saveToVideo(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        plt.figure(4)
        plt.clf()
        ax = plt.axes()
        ax.set_facecolor("grey")
        x = tf.random.uniform((80_000,), -2.0, 0.7, tf.float32)
        y = tf.random.uniform((80_000,), -1.3, 1.3, tf.float32)
        data = tf.stack([x, y], axis=1)
        predictions = model.predict(data)
        pred2 = tf.transpose(predictions)[1]
        # pred3 = tf.math.log(tf.add(pred2, tf.abs(tf.reduce_min(pred2))))
        pred3 = tf.math.log_sigmoid(pred2)
        plot = plt.scatter(x, y, s=1, c=pred3)
        plt.savefig("captures/autosave.png")
        im = imageio.imread("captures/autosave.png")
        writer.append_data(im)
        writer.append_data(im) # add a copy of the frame to slow down the playback

#%%

class MandelSequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size, batch_per_seq):
        self.batch_size = batch_size
        self.batch_per_seq = batch_per_seq
    def __len__(self):
        return self.batch_per_seq
    def __getitem__(self, item):
        return MandelbrotDataSet(self.batch_size)

#%%
start = time.time()

BATCH_SIZE = 1024
BATCH_PER_SEQ = 64
EPOCHS = 128
LR = 0.002

HIDDENLAYERS = 10
LAYERWIDTH = 126    #126,288s ; 1024, 694s

# CREATE MODEL
# ------------
model = tf.keras.Sequential()

# ADD LAYER
# ---------

# input
tf.keras.Input(shape=(2,))

# hidden layers
for _ in range(HIDDENLAYERS):
    model.add(tf.keras.layers.Dense(LAYERWIDTH, activation="gelu"))

# output
model.add(tf.keras.layers.Dense(2, activation=None))

# compile model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
#              optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0),
              metrics=["accuracy", "mae", "mse"])

# init sequence generator
sequence = MandelSequence(BATCH_SIZE, BATCH_PER_SEQ)
# train (simple)
history = model.fit(sequence,epochs=EPOCHS)

# train (with validation)
#val_sequence = MandelbrotDataSet(BATCH_SIZE)
#history = model.fit(sequence,epochs=EPOCHS, validation_data=(val_sequence.data, val_sequence.outputs))

# train (with video callback)
# history = model.fit(sequence, epochs=EPOCHS, callbacks=[saveToVideo()])

#print("Evaluate on test data")
#eval_sequence = MandelSequence(BATCH_SIZE, 2)
#results = model.evaluate(eval_sequence)
#print("test loss, test acc:", results)

#%%

# VISUALIZATION
# -------------

np.set_printoptions(precision=3, suppress=True)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
plt.figure(1)
def plot_loss(history):
#  plt.plot(history.history['val_loss'], label='validation loss')
  plt.plot(history.history['loss'], label='training loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.ylim((0.0, 0.2))
  plt.legend()
  plt.grid(True)

plot_loss(history)

#%%
x = tf.random.uniform((200_000,), -2.0, 0.7, tf.float32)
y = tf.random.uniform((200_000,), -1.3, 1.3, tf.float32)
data = tf.stack([x, y], axis=1)
predictions = model.predict(data)
pred2 = tf.transpose(predictions)[1]
pred3 = tf.math.sqrt(tf.add(pred2, tf.abs(tf.reduce_min(pred2))))
pred4 = tf.math.log_sigmoid(pred2)
#%%
plt.figure(2, figsize=(2.7*3, 2.6*3), dpi=300)
plot = plt.scatter(x, y, s=1, c=pred3)
plt.figure(21, figsize=(2.7*3, 2.6*3), dpi=300)
plot = plt.scatter(x, y, s=1, c=pred4)
plt.show()

#%%

#save model
ts = int(time.time())
file_path = f"models/mandelbrain-{ts}/"
model.save(filepath=file_path, save_format='tf')

writer.close()

print("elapsed : ", time.time() - start)
