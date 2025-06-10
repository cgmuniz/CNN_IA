import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import keras_tuner
import matplotlib.pyplot as plot
import seaborn as sb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Conv2D(
        filters=hp.Int('filters_0', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size_0', values=[3, 5]),
        activation=hp.Choice("activation", ["relu", "tanh"]),
        input_shape=(28, 28, 1)
    ))
    model.add(layers.MaxPooling2D(pool_size=hp.Choice('pool_size_0', values=[2, 3])))
    for i in range(1, hp.Int('conv_layers', 1, 3)):
        model.add(layers.Conv2D(
            filters=hp.Int(f'filters_{i}', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice(f'kernel_size_{i}', values=[3, 5]),
            activation=hp.Choice("activation", ["relu", "tanh"])
        ))
        model.add(layers.MaxPooling2D(pool_size=hp.Choice(f'pool_size_{i}', values=[2, 3])))
    model.add(layers.Flatten())
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.05)))
    model.add(layers.Dense(10, activation="softmax"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

tuner = keras_tuner.Hyperband(build_model,
                    objective='val_accuracy',
                    max_epochs=10
                    )

(x_mc, y_mc), (x_mc_test, y_mc_test) = keras.datasets.mnist.load_data()

runs = [[x_mc, y_mc, x_mc_test, y_mc_test]]

train_filter = np.where((y_mc == 0) | (y_mc == 1))
test_filter = np.where((y_mc_test == 0) | (y_mc_test == 1))

x_bin, y_bin = x_mc[train_filter], y_mc[train_filter]
x_bin_test, y_bin_test = x_mc_test[test_filter], y_mc_test[test_filter]

runs.append([x_bin, y_bin, x_bin_test, y_bin_test])

for run in runs:
    x = run[0]
    y = run[1]
    x_test = run[2]
    y_test = run[3]

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=42)

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
    x_val = x_val.reshape(-1, 28, 28, 1).astype('float32')
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')

    x_train /= 255.0
    x_val /= 255.0
    x_test /= 255.0

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    stop_early = tf.keras.callbacks.EarlyStopping(patience=5)

    tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val), verbose=1, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters()[0]
    print(f"Melhores hiperparâmetros: {best_hps.values}")

    model = build_model(best_hps)
    initial_weights = [layer.get_weights() for layer in model.layers]
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), verbose=1, callbacks=[stop_early])

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Melhor época: %d' % (best_epoch,))

    model = build_model(best_hps)
    x_all = np.concatenate((x_train, x_val))
    y_all = np.concatenate((y_train, y_val))
    model.fit(x=x_all, y=y_all, epochs=best_epoch)

    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    final_weights = [layer.get_weights() for layer in model.layers]

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    plot.figure(figsize=(10, 8))
    sb.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plot.xlabel('Predicted')
    plot.ylabel('True')
    plot.title('Confusion Matrix')
    plot.show()

    # Plotar o gráfico de loss do treinamento
    plot.figure(figsize=(10, 6))
    plot.plot(history.history['loss'], label='Training Loss')
    plot.plot(history.history['val_loss'], label='Validation Loss')
    plot.xlabel('Epochs')
    plot.ylabel('Loss')
    plot.legend()
    plot.title('Training and Validation Loss')
    plot.show()

    # Salvar hiperparâmetros e pesos

    hyperparameters = model.summary()
    print(hyperparameters)

    # Salvar hiperparâmetros
    with open('hyperparameters.pkl', 'wb') as f:
        pickle.dump(hyperparameters, f)

    # Salvar pesos iniciais
    with open('initial_weights.pkl', 'wb') as f:
        pickle.dump(initial_weights, f)

    # Salvar pesos finais
    with open('final_weights.pkl', 'wb') as f:
        pickle.dump(final_weights, f)

    # Salvar histórico do treinamento (erro por iteração)
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Salvar as saídas da rede neural para os dados de teste
    with open('test_predictions.pkl', 'wb') as f:
        pickle.dump(y_pred, f)