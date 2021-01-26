import tensorflow as tf
from tensorflow.data import Dataset


def compile_and_fit(model,
                    train_data: Dataset,
                    validation_data: Dataset,
                    loss=tf.losses.MeanSquaredError(),
                    epochs=100,
                    optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9),
                    callbacks: list = [],
                    metrics: list = []):
    """"Compiles and fits model. Prints model summary and returns learning history"""

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    print(model.summary())

    history = model.fit(train_data, epochs=epochs,
                        validation_data=validation_data,
                        verbose=0,
                        callbacks=callbacks)
    return history


def apply_model(model, test: Dataset):
    y_predicted = model.predict(test)
    y_predicted = y_predicted.flatten()
    # extract labels into array
    y_true = tf.concat([batch for batch in test.map(lambda feature, label: label)], axis=0)
    y_true = y_true.numpy().flatten()
    mse = tf.keras.metrics.mean_absolute_error(y_true, y_predicted).numpy()
    return y_true, y_predicted,  mse
