import tensorflow as tf

def main():
    x = tf.constant(4)
    
    with tf.GradientTape() as grad:
        grad.watch(x)
        y = tf.math.sin(x)
    dy_dx = grad.gradient(y, x)
    print(dy_dx)

if __name__ == '__main__':
    main()