import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, Conv1D, Embedding, MaxPooling1D, Flatten, Input, concatenate, Bidirectional, \
    LSTM, SimpleRNN
from keras.models import Sequential
from keras import layers, Model
import numpy as np



class RNN_Branch(Model):


    def __init__(self):
        super(RNN_Branch, self).__init__()
        self.rnn_embeddings = Embedding(5, output_dim=100,input_length=130)
        self.bidirectional = Bidirectional(LSTM(256, return_sequences=False,activation="relu"))
        # self.lstm = LSTM(256, name="lstm_layer")
        self.ann_dense = Dense(10, activation='relu')
        self.ann_output = Dense(1, activation='sigmoid')

    def call(self, inputs, training=False, **kwargs):
        x = self.rnn_embeddings(inputs)
        x = self.bidirectional(x)
        # x = self.lstm(x)
        x = self.ann_dense(x)
        x = self.ann_output(x)
        return x



class CNN_Branch(layers.Layer):


    def __init__(self):
        super(CNN_Branch, self).__init__()
        self.cnn_embeddings = Embedding(5, output_dim=100,input_length=130)
        self.cnn_conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')
        self.cnn_maxpool = MaxPooling1D(pool_size=2)
        self.cnn_flatten = Flatten()
        self.cnn_dense = Dense(10, activation='relu')
        self.cnn_output = Dense(1, activation='sigmoid')

    def call(self, inputs, training=False, **kwargs):
        x = self.cnn_embeddings(inputs)
        x = self.cnn_conv1(x)
        x = self.cnn_flatten(x)
        x = self.cnn_dense(x)
        x = self.cnn_output(x)
        return x


class ANN_Branch(Model):


    def __init__(self):
        super(ANN_Branch, self).__init__()

        self.ann_dense1 = Dense(100, activation='relu')
        self.ann_dense2 = Dense(50, activation='relu')
        self.ann_dense3 = Dense(20, activation='relu')
        self.ann_dropout = Dropout(rate=0.5)
        self.ann_output = Dense(1, activation='sigmoid')


    def call(self, inputs, training=False, **kwargs):
        x = self.ann_dense1(inputs)
        x = self.ann_dense2(x)
        x = self.ann_dropout(x)
        x = self.ann_dense3(x)
        x = self.ann_output(x)
        return x

    def model(self):
        x1 = keras.Input(shape=(587, ))
        return keras.Model(inputs=x1, outputs=self.call(x1))


class miTransfer(Model):

    def __init__(self):
        super(miTransfer, self).__init__()
        self.ann_block = ANN_Branch()
        self.cnn_block = CNN_Branch()
        self.rnn_block = RNN_Branch()
        self.combined_layer = Dense(2, activation="relu")
        self.final_output = Dense(1, activation='sigmoid')


    def call(self, inputs, **kwargs):
        ann_output= self.ann_block(inputs[0])
        cnn_output = self.cnn_block(inputs[1])
        rnn_output = self.rnn_block(inputs[1])
        combined_input = concatenate([ann_output, cnn_output,rnn_output])
        x = self.combined_layer(combined_input)
        return self.final_output(x)


    # def model(self):
    #     x1 = keras.Input(shape=(587, ))
    #     x2 = keras.Input(shape=(587, ))
    #     return keras.Model(inputs=[x1,x2], outputs=self.call([x1,x2]))
    #


# model = CNN_Branch()
# model = RNN_Branch()

# model = ANN_Branch().model()
model = miTransfer()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

# input_data = tf.random.normal((1504, 587))
input_data = np.random.random([1504, 587]) * 5
input_data = input_data.astype('int')
print(input_data.shape)
# input_data = np.array([[1, 2, 3], [1, 1, 1], [1, 1, 1]])
# print(input_data.shape)
pred = model.predict([input_data,input_data])
# pred = model.predict(input_data)
print(model.summary())
print(input_data.shape)
print(pred.shape)

model.save('delete_model')

i=0