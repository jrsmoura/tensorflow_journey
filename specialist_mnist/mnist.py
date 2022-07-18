import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation="relu")
        self.d2 = tf.keras.layers.Dense(10)

    def call(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

@tf.function
def train_step(images, labels,
               model: tf.keras.Model,
               loss_object: tf.keras.losses,
               optimizer: tf.keras.optimizers) -> None:
    """Function for one training step.

    Args:
        images (_type_): _description_
        labels (_type_): _description_
        model (tf.keras.Model): _description_
        loss_object (tf.keras.losses): _description_
        optimizer (tf.keras.optimizers): _description_
    """
    with tf.GradientTape() as tape:
        prediction = model(images, training=True, labels=labels)
        loss = loss_object(labels, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def test_step(images, labels,
              model: tf.keras.Model,
              loss_object: tf.keras.losses,
              test_loss: tf.keras.metrics,
              test_accuracy: tf.keras.metrics) -> None:
    """Function for model test. One step.

    Args:
        images (_type_): _description_
        labels (_type_): _description_
        model (tf.keras.Model): _description_
        loss_object (tf.keras.losses): _description_
        test_loss (tf.keras.Metrics): _description_
        test_accuracy (tf.keras.metrics): _description_
    """    
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


def main():
    # Loading data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize data features
    x_train, x_test = x_train / 255., x_test / 255.

    # We are working with image data, so we need an extra dimension for chanel
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    # Use tf.data to aggregate and shuffle dataset
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).shuffle(10000).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    model = MyModel()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()
     
    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    EPOCHS = 5
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels, model, loss_object, optimizer)

        for test_images, test_labels in test_ds:
            test_step(images, labels, model, loss_object, test_loss, test_accuracy) 
        
        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
  

if __name__ == '__main__':
    main()
