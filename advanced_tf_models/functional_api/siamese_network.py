import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
import random
import numpy as np


def initialize_base_network():
    """Build a base network for a siamese network
    """
    input_layer = Input(shape=(28, 28,), name="input_layer")
    x = Flatten()(input_layer)
    x = Dense(units="128", activation=tf.nn.relu,
              name="first_dense")(input_layer)
    x = Dropout(0.1, name="first_droptout")(x)
    x = Dense(units=128, activation=tf.nn.relu, name="second_dense")(x)
    x = Dropout(0.1, name="second_doupout")(x)
    x = Dense(units="128", activation=tf.nn.relu, name="third_dense")(x)

    return Model(inputs=input_layer, outputs=x)


def create_pairs(x, digit_indices):
    """Positive and negative pair creation.
    Alternates between positive and negative pairs.
    """
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1

    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randomrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]

    return np.array(pairs), np.array(labels)


def create_pairs_on_set(images, labels):
    digit_indices = [np.where(labels == i)[0] for i in range(10)]
    pairs, y = create_pairs(images, digit_indices)
    y = y.astype("float32")

    return pairs, y


def main():
    (train_img, train_lb), (test_img, test_lb) = fashion_mnist.load_data()
    train_img = train_img / 255.
    test_img = test_img / 255.


if __name__ == "__main__":
    main()
