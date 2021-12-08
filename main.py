#%%
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import imageio

print(tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
#print(tf.executing_eagerly())

# Hide GPU from visible devices
#tf.config.set_visible_devices([], 'GPU')
#tf.debugging.set_log_device_placement(True)

#%%

#class MandelbrotDataSet:
@tf.function
def MandelbrotDataSet(size=1000, max_depth=100, xmin=-2.0, xmax=0.7, ymin=-1.3, ymax=1.3):
    x = tf.random.uniform((size,),xmin,xmax,tf.bfloat16)
    y = tf.random.uniform((size,),ymin,ymax,tf.bfloat16)
    return tf.stack([x, y], axis=1), mandel(x=x, y=y,max_depth=max_depth)

@tf.function
def mandel(x, y, max_depth):
    zx, zy = x,y
    for n in range(1, max_depth):
        zx, zy = zx*zx - zy*zy + x, 2*zx*zy + y
    return tf.cast(tf.less(zx*zx+zy*zy, 4.0),tf.float16) #* 2.0 - 1.0

#mds = tf.function(MandelbrotDataSet)
#print(tf.autograph.to_code(mds))

#%%
# VIDEO
writer = imageio.get_writer('./captures/autosave.mp4', fps=30)
#capture_rate=10

#%%
class saveToVideo(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        plt.figure(4)
        plt.clf()
        ax = plt.axes()
        ax.set_facecolor("black")
        x = tf.random.uniform((40_000,), -2.0, 0.7, tf.float16)
        y = tf.random.uniform((40_000,), -1.3, 1.3, tf.float16)
        data = tf.stack([x, y], axis=1)
        predictions = model.predict(data)
        plot = plt.scatter(x, y, s=1, c=predictions)
        plt.savefig("captures/autosave.png")
        writer.append_data(imageio.imread("captures/autosave.png"))

#%%

class MandelSequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size, batch_per_seq):
        self.batch_size = batch_size
        self.batch_per_seq = batch_per_seq
    def __len__(self):
        return self.batch_per_seq
    def __getitem__(self, item):
#        batch = MandelbrotDataSet(self.batch_size)
#        return batch.data, batch.outputs
        return MandelbrotDataSet(self.batch_size)

#plt.figure(3)
#mb1, mb2 = MandelbrotDataSet(100_000)
#plot = plt.scatter(mb1[0], mb1[1], s=1, c=mb2)
#plt.show()
#exit(1)
#%%

BATCH_SIZE = 8192
BATCH_PER_SEQ = 25
EPOCHS = 400
LR = 0.0018

HIDDENLAYERS = 10
LAYERWIDTH = 256

model = tf.keras.Sequential()
tf.keras.Input(shape=(2,))

for _ in range(HIDDENLAYERS):
    model.add(tf.keras.layers.Dense(LAYERWIDTH, activation="gelu"))

model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
#              optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0),
              metrics=["accuracy", "mae", "mse"])

#%%

sequence = MandelSequence(BATCH_SIZE, BATCH_PER_SEQ)
#val_sequence = MandelbrotDataSet(BATCH_SIZE)
#input, output = MandelbrotDataSet(100_000)
history = model.fit(sequence,epochs=EPOCHS)#, #validation_data=(val_sequence.data, val_sequence.outputs),
#history = model.fit(input, output,epochs=EPOCHS, batch_size=BATCH_SIZE)#, #validation_data=(val_sequence.data, val_sequence.outputs),
#                    callbacks=[saveToVideo()])
#                    callbacks=[])

#%%
#print("Evaluate on test data")
#eval_sequence = MandelSequence(BATCH_SIZE, 2)
#results = model.evaluate(eval_sequence)
#print("test loss, test acc:", results)

#%%

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
  plt.ylim((0.0, 0.04))
  plt.legend()
  plt.grid(True)

plot_loss(history)

#%%
x = tf.random.uniform((200_000,), -2.0, 0.7, tf.float16)
y = tf.random.uniform((200_000,), -1.3, 1.3, tf.float16)
data = tf.stack([x, y], axis=1)
predictions = model.predict(data)

#%%
plt.figure(2, figsize=(2.7*3, 2.6*3), dpi=300)
plot = plt.scatter(x, y, s=1, c=predictions)
plt.show()

#%%

#save model
ts = int(time.time())
file_path = f"models/mandelbrain-{ts}/"
model.save(filepath=file_path, save_format='tf')

writer.close()
