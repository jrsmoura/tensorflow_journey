import tensorflow as tf

def main():
    # We initialize a tf.Variable given it a shape and type
    # that can not be changed, but the values can be.

    x = tf.Variable(22, dtype=tf.int32, name="my_var")
    print(x)
    # We can change the value using one of the assign methods
    x_2 = x.assign(23)
    print(f'original: {x}\nnew: {x_2}')

    # Remember ... a variable is a tensor whose value can be changed ..

    w = tf.Variable([[1., 2.], [2., 1.]])
    x = tf.constant([[3., 4.],
                     [1., 3.]])

    print(w.shape)
    print(tf.matmul(w, x))
    print(w @ x)

    xx = tf.zeros((3, 4))
    print(xx)

if __name__ == '__main__':
    main()