import os 
import tensorflow as tf
import pandas as pd
from tensorflow import keras

path = "/checkpoints"
checkpoint_path = "./checkpoints"
# saves trained model evry epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=1)
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
initializer = tf.keras.initializers.VarianceScaling()

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0


model = keras.Sequential([
    # input layer
    keras.layers.Flatten(input_shape=(28, 28)),
    # first hidden layer
    keras.layers.Dense(200, activation='elu', kernel_initializer=initializer),
    # dropout function 
    tf.keras.layers.Dropout(0.10),
    # second hiddn layer
    keras.layers.Dense(100, activation='elu', kernel_initializer=initializer),
    # batch normalization
    keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001),
    # output layer
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=adam,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              )

model.fit(x_train, y_train, epochs=10, callbacks=[cp_callback])
test_loss, test_acc = model.evaluate(x_test, y_test)


print("Predictions accurasy is", test_acc)

