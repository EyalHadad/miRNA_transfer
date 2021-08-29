from keras.layers import Dense,Dropout,Conv1D,GlobalMaxPooling1D,Embedding,MaxPooling1D,Flatten,LSTM,Input,concatenate
from keras.models import Sequential,Model
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.regularizers import l1




def base_learning(my_shape):
    print("----- 4 Layers network------")
    model = Sequential()
    model.add(Dense(my_shape, input_dim=my_shape, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(my_shape/2, activation='relu',kernel_constraint=maxnorm(3),activity_regularizer=l1(0.001),kernel_regularizer=l1(0.001)))
    model.add(Dropout(rate=0.5))
    model.add(Dense(my_shape/4, activation='relu',kernel_constraint=maxnorm(3),activity_regularizer=l1(0.001),kernel_regularizer=l1(0.001)))
    model.add(Dense(my_shape/8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['acc'])
    return model



def cnn_learning2():
    # define model
    model = Sequential()
    model.add(Embedding(5, 100, input_length=130))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    print(model.summary())

    return model



def rnn_learning():
    # define model
    model = Sequential()
    model.add(Embedding(5, 100, input_length=130))
    model.add(LSTM(100))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    print(model.summary())

    return model


def build_cnn_branch():
    inputs = Input(shape=(130,))
    y = Embedding(5, 100)(inputs)
    y = Conv1D(filters=32, kernel_size=8, activation='relu')(y)
    y = MaxPooling1D(pool_size=2)(y)
    y = Flatten()(y)
    y = Dense(10, activation='relu')(y)
    y = Dense(1, activation='sigmoid')(y)
    y = Model(inputs, y)
    return y


def miTrans_base_model(my_shape):
    print("----- 4 Layers network------")

    ann_branch = build_ann_branch(my_shape)
    cnn_branch = build_cnn_branch()
    combined = concatenate([ann_branch.output, cnn_branch.output])
    z = Dense(2, activation="relu")(combined)
    z = Dense(1, activation = 'sigmoid')(z)
    model = Model(inputs=[ann_branch.input, cnn_branch.input], outputs=z)
    # opt = Adam(lr=1e-3, decay=1e-3 / 200)
    # model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=['acc'])
    print(model.summary())
    return model


def build_ann_branch(my_shape):
    inputs = Input(shape=(my_shape,))
    x = Dense(my_shape, activation='relu')(inputs)
    x = Dense(my_shape / 2, activation='relu', kernel_constraint=maxnorm(3), activity_regularizer=l1(0.001),
              kernel_regularizer=l1(0.001))(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(my_shape / 4, activation='relu', kernel_constraint=maxnorm(3), activity_regularizer=l1(0.001),
              kernel_regularizer=l1(0.001))(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(my_shape / 8, activation='relu', kernel_constraint=maxnorm(3), activity_regularizer=l1(0.001),
              kernel_regularizer=l1(0.001))(x)
    x = Dense(1, activation='sigmoid')(x)
    x = Model(inputs, x)
    return x


