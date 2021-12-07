#%%
import time

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

print(tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

#%%

class MandelbrotDataSet:
    def __init__(self, size=1000, max_depth=100, xmin=-2.0, xmax=0.7, ymin=-1.3, ymax=1.3):
        self.x = tf.random.uniform((size,),xmin,xmax,tf.float16)
        self.y = tf.random.uniform((size,),ymin,ymax,tf.float16)
        self.outputs = self.mandel(x=self.x, y=self.y,max_depth=max_depth)
        self.data = tf.stack([self.x, self.y], axis=1)

    @staticmethod
    def mandel(x, y, max_depth):
        zx, zy = x,y
        for n in range(1, max_depth):
            zx, zy = zx*zx - zy*zy + x, 2*zx*zy + y
        return tf.cast(tf.less(zx*zx+zy*zy, 4.0),tf.float16) #* 2.0 - 1.0

#%%

#trainingData = MandelbrotDataSet(200_000)

class MandelSequence(tf.keras.utils.Sequence):
    def __init__(self, batch_size, batch_per_seq):
        self.batch_size = batch_size
        self.batch_per_seq = batch_per_seq
    def __len__(self):
        return self.batch_per_seq
    def __getitem__(self, item):
        batch = MandelbrotDataSet(self.batch_size)
        return batch.data, batch.outputs


BATCH_PER_SEQ = 100
HIDDENLAYERS = 10
LAYERWIDTH = 60
LR = 0.0012
EPOCHS = 50
BATCH_SIZE = 1000

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
val_sequence = MandelSequence(BATCH_SIZE, 2)
history = model.fit(sequence,epochs=EPOCHS, validation_data=val_sequence)

#%%
print("Evaluate on test data")
eval_sequence = MandelSequence(BATCH_SIZE, 2)
results = model.evaluate(eval_sequence)
print("test loss, test acc:", results)

#%%

np.set_printoptions(precision=3, suppress=True)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
plt.figure(1)
def plot_loss(history):
  plt.plot(history.history['loss'], label='training loss')
  plt.plot(history.history['val_loss'], label='validation loss')
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
plt.figure(2)
plot = plt.scatter(x, y, s=1, c=predictions)
plt.show()

#%%

#save model
ts = int(time.time())
file_path = f"models/mandelbrain-{ts}/"
model.save(filepath=file_path, save_format='tf')
