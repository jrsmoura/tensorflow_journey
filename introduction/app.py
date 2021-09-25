import tensorflow as tf


def main():
    # 0d tenosr, AKA scalar
    x_0 = tf.constant(1, dtype=tf.int32)
    print(x_0.shape)
    # 1d tenosr, AKA vector
    x_1 = tf.constant([1, 2, 3],dtype=tf.int32)
    print(x_1.shape)
    # 2d tenosr, AKA matrix
    x_2 = tf.constant([[1, 2, 3],
                       [4, 5, 6]], dtype=tf.int32)
    print(x_2.shape)
    # 3d tenosr, AKA tensor ... from here ... anyone is a tensor
    x_3 = tf.constant([
                        [[1, 2, 3], [4, 5, 6]],
                        [[7, 8, 9], [0, 9, 8]]
                    ])
    print(x_3.shape)

    # Reshape: x_2.shape = (2, 3) and we want to reshape it to (3, 2)
    x_2_respahe = tf.reshape(x_2, [3, 2])
    print(x_2_respahe.shape)

if __name__ == '__main__':
    main()