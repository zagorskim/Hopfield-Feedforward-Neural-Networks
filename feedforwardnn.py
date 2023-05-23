import numpy as np
from tensorflow import keras


def feedforward_train(x_train, y_train, x_test, y_test, epochs):
    model_feedforward = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(x_train.shape[1], x_train.shape[2])),
        keras.layers.Dense(x_train.shape[1] * x_train.shape[2], activation='relu'),
        keras.layers.Dropout(0.2, name='layers_dropout'),
        keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
    ])
    logdir = "logs/fit"
    tensorboard_callback = keras.callbacks.TensorBoard(histogram_freq=1, log_dir=logdir)
    best_model_callback = keras.callbacks.ModelCheckpoint(filepath=logdir + "/checkpoint",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
    model_feedforward.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
    model_feedforward.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test),
                          callbacks=[tensorboard_callback, best_model_callback])
    model_feedforward.load_weights(logdir + "/checkpoint")
    y_pred = model_feedforward.predict(x_test)
    pred_labels = np.argmax(y_pred, axis=1)
    matching_count = np.count_nonzero(pred_labels == y_test)
    return matching_count / len(pred_labels)



