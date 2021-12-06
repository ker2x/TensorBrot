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
        return tf.cast(tf.less(zx*zx+zy*zy, 4.0),tf.float16)# * 2.0 - 1.0

#%%

trainingData = MandelbrotDataSet(1_000_000)

#%%
#plt.figure(3)
#plt.scatter(trainingData.x, trainingData.y, s=1, c=trainingData.outputs)
#plt.show()
#%%

HIDDENLAYERS = 10
LAYERWIDTH = 20
LR = 0.0008
EPOCHS = 100
BATCH_SIZE = 1000

model = tf.keras.Sequential()
tf.keras.Input(shape=(2,)),

#model.add(tf.keras.layers.Dense(LAYERWIDTH, activation="gelu"))
#model.add(tf.keras.layers.Dense(LAYERWIDTH, activation="gelu"))
#model.add(tf.keras.layers.Dense(4, activation="gelu"))

for _ in range(HIDDENLAYERS):
    model.add(tf.keras.layers.Dense(LAYERWIDTH, activation="gelu"))

model.add(tf.keras.layers.Dense(2, activation="gelu"))
model.add(tf.keras.layers.Dense(1,activation=None))

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
#              optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0),
              metrics=["accuracy", "mae", "mse"])

#%%

history = model.fit(trainingData.data, trainingData.outputs,epochs=EPOCHS,batch_size=BATCH_SIZE,
                    shuffle=True)

#%%

np.set_printoptions(precision=3, suppress=True)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
plt.figure(1)
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.ylim((0.0, 0.1))
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
