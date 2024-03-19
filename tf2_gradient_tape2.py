# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.losses import MSE
import tensorflow as tf
import numpy as np
import time
import sys

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#from tensorflow.python.framework.ops import disable_eager_execution

#disable_eager_execution()

x = tf.constant(3.0)
y = tf.constant(2.0)
with tf.GradientTape(persistent=True) as h:
    h.watch(x)
    h.watch(y)
    with tf.GradientTape(persistent=True) as g:
        g.watch(x)
        g.watch(y)
        f = 2*x**2 - x*y + 3*y**2
    fx = g.gradient(f, x)
    fy = g.gradient(f, y)
fxx = h.gradient(fx, x)
fxy = h.gradient(fx, y)
fyx = h.gradient(fy, x)
fyy = h.gradient(fy, y)

print(fx.numpy())
print(fy.numpy())
print(fxx.numpy())
print(fxy.numpy())
print(fyx.numpy())
print(fyy.numpy())

def build_model(D):
    #Input(shape = (D,))
	# build the model using Keras' Sequential API
	model = Sequential([
	    Dense(D, activation='relu'),
        Dense(D, activation='relu')
    ])

	# return the built model to the calling function
	return model

D=2
N=4
model = build_model(D)
X=tf.random.uniform(shape=[N,D], dtype=tf.float32)
y=tf.random.uniform(shape=[N,D], dtype=tf.float32)

with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        pred = model(X)
        f = MSE(y, pred)
    g = tape1.gradient(f, model.trainable_variables)
    flat_g = tf.concat([tf.reshape(grad, [-1]) for grad in g], axis=0) #needed to avoid list type not having shape.
h = tape2.jacobian(flat_g, model.trainable_variables)

print(g)
print(h)
