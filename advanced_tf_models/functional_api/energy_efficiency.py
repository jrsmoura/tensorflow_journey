import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def format_output(data):
    y1 = np.array(data.pop('Y1'))
    y2 = np.array(data.pop('Y2'))
    return y1, y2


def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']


def plot_diff(y_true, y_pred, title=''):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.show()


def plot_metrics(metric_name, history, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name],
             color='green', label='val_' + metric_name)
    plt.show()


def generate_data_stats(data):
    data_stats = data.describe()
    data_stats.pop("Y1")
    data_stats.pop("Y2")
    data_stats = data_stats.transpose()
    return data_stats


def build_model(train: pd.DataFrame) -> Model:
    input_layer = Input(shape=len(train.columns),
                        name="input_layer")
    first_dense = Dense(units='128',
                        activation=tf.nn.relu,
                        name="first_dense")(input_layer)
    second_dense = Dense(units='128',
                         activation=tf.nn.relu,
                         name="second_dense")(first_dense)
    y1_output = Dense(units='1',
                      name='y1_output')(second_dense)
    third_dense = Dense(units='64',
                        activation=tf.nn.relu,
                        name="third_dense")(second_dense)
    y2_output = Dense(units='1',
                      name='y2_output')(third_dense)

    return Model(inputs=input_layer,
                 outputs=[y1_output, y2_output])


def main():
    df = pd.read_excel("../data/ENB2012_data.xlsx")
    df = df.sample(frac=1).reset_index(drop=True)

    train, test = train_test_split(df, test_size=0.2)
    train_stats = generate_data_stats(train)

    train_Y = format_output(train)
    test_Y = format_output(test)

    norm_train_X = norm(train, train_stats=train_stats)
    norm_test_X = norm(test, train_stats=train_stats)

    model = build_model(train=train)
    print(model.summary())

    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss={"y1_output": "mse",
                        "y2_output": "mse"},
                  metrics={"y1_output": tf.keras.metrics.RootMeanSquaredError(),
                           "y2_output": tf.keras.metrics.RootMeanSquaredError(), }
                  )

    history = model.fit(norm_train_X, train_Y,
                        epochs=500,
                        batch_size=10,
                        validation_data=(norm_test_X, test_Y))
    loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = model.evaluate(x=norm_test_X,
                                                              y=test_Y)
    print("Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(loss,
                                                                                   Y1_loss,
                                                                                   Y1_rmse,
                                                                                   Y2_loss,
                                                                                   Y2_rmse))

    Y_pred = model.predict(norm_test_X)
    plot_diff(test_Y[0], Y_pred[0], title='Y1')

    plot_diff(test_Y[1], Y_pred[1], title='Y2')
    plot_metrics(metric_name='y1_output_root_mean_squared_error',
                 history=history,
                 title='Y1 RMSE', ylim=6)
    plot_metrics(metric_name='y2_output_root_mean_squared_error',
                 history=history,
                 title='Y2 RMSE', ylim=7)


if __name__ == "__main__":
    main()
