import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from tensorflow.keras.models import Model


def initialize_base_network():
    input_layer = Input(shape=(28, 28,), name="input_layer")
    x = Flatten()(input_layer)
    x = Dense(units="128", activation=tf.nn.relu,
              name="first_dense")(input_layer)
    x = Dropout(0.1, name="first_droptout")(x)
    x = Dense(units=128, activation=tf.nn.relu, name="second_dense")(x)
    x = Dropout(0.1, name="second_doupout")(x)
    x = Dense(units="128", activation=tf.nn.relu, name="third_dense")(x)

    return Model(inputs=input_layer, outputs=x)
