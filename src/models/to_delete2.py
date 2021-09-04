import tensorflow as tf
from tensorflow import keras as K
from keras.layers import Dense, Dropout, Conv1D, Embedding, MaxPooling1D, Flatten, Input, concatenate,Conv2D,MaxPooling2D
from keras import layers, Model
import numpy as np
from keras.models import Sequential


class CNN_Branch(Model):
    def get_config(self):
        pass

    def __init__(self, vocab_size=500, embed_dim=1000):
        super(CNN_Branch, self).__init__()
        # self.cnn_input = Input(shape=(input_shape,))
        self.cnn_embeddings = Embedding(vocab_size, embed_dim)
        self.cnn_conv1 = Conv1D(filters=32, kernel_size=8, activation='relu')
        self.cnn_maxpool = MaxPooling1D(pool_size=2)
        self.cnn_flatten = Flatten()
        self.cnn_dense = Dense(10, activation='relu')
        self.cnn_output = Dense(1, activation='sigmoid')

    def call(self, inputs, training=False, **kwargs):
        # x = self.cnn_embeddings(inputs)
        print("First:",inputs.get_shape())
        x = self.cnn_conv1(inputs)
        print("Second:",x.get_shape())

        x = self.cnn_maxpool(x)
        print("Third:",x.get_shape())
        x = self.cnn_flatten(x)
        print("Fourth:",x.get_shape())
        x = self.cnn_dense(x)
        x = self.cnn_output(x)
        return x


model = CNN_Branch()
# input_data = np.array([[1,2,3],[1,1,1],[1,1,1]])
# input_data = np.array([1,2,3])

input_data = tf.random.normal((1504, 130, 1))
pred = model.predict(input_data)
print(input_data.shape)
print(pred.shape)

i=9