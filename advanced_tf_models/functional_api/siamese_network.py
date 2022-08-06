import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import backend as K
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

def euclidian_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


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
            inc = random.randrange(1, 10)
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

def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def main():
    (train_img, train_lb), (test_img, test_lb) = fashion_mnist.load_data()
    train_img = train_img / 255.
    test_img = test_img / 255.
    tr_pairs, tr_y = create_pairs_on_set(train_img, train_lb)
    ts_pairs, ts_y = create_pairs_on_set(test_img, test_lb)

    # pequeno teste de exibição
    this_pair = 8
    show_image(ts_pairs[this_pair][0])
    show_image(ts_pairs[this_pair][1])

if __name__ == "__main__":
    main()
