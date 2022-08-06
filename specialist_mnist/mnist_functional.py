import tensorflow as tf
from typing import tuple, list

"""MNIST classification problem using functional programming

STEPS: 
    1. load data from tf.keras.datasets.mnist
    2. split train/validation datasets
    3. statarize data
    4. adjust input dimensions
    5. define architecture: [conv2d] -> [flatten] -> [dense] -> [dense]
    6. setup loss object
    7. setup optimizer object
    8. setup metric for training and validation
"""

def fetch_data() -> tuple:
    """Fetch MNIST data from tf.keras.datasets

    Returns:
        tuple: two tuples: (x_train, y_train), (x_val, y_val)
    """
    return tf.keras.datasets.mnist.load_data()





def main():
    ...


if __name__ == '__main__':
    main()