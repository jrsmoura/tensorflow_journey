import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Sequential, Model
# from tensorflow.python.keras.utils.vis_utils import plot_model


def build_mode_with_sequential():
    sq_model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=tf.nn.relu),
        Dense(10, activation=tf.nn.softmax)
    ])
    return sq_model


def build_model_with_functional():
    input_layer = Input(shape=(28, 28))
    flatten_layer = Flatten()(input_layer)
    x = Dense(128, activation=tf.nn.relu)(flatten_layer)
    output_layer = Dense(10, activation=tf.nn.softmax)(x)

    func_model = Model(inputs=input_layer, outputs=output_layer)
    return func_model


def main():
    model = build_model_with_functional()
#    plot_model(model, show_shapes=True, show_layer_names=True, to_file="model.png")
    print(model.summary())

    # fetch mnist data
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    # normalize data
    training_images = training_images / 255.
    test_images = test_images / 255.
    
    # configure, train and evaluate the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(training_images, training_labels, epochs=5)
    eval = model.evaluate(test_images, test_labels)
    print(eval)

if __name__ == "__main__":
    main()
