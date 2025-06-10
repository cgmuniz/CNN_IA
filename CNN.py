import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import keras_tuner
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import json

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(28, 28, 1)))
    model.add(layers.Conv2D(
        filters=hp.Int('filters_0', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size_0', values=[3, 5]),
        activation=hp.Choice("activation", ["relu", "tanh"])
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

# Carregar os dados do MNIST
(x_mc, y_mc), (x_mc_test, y_mc_test) = keras.datasets.mnist.load_data()

# Definir run multiclasses
runs = [[x_mc, y_mc, x_mc_test, y_mc_test]]

# Criar um filtro para o caso binário
train_filter = np.where((y_mc == 0) | (y_mc == 1))
test_filter = np.where((y_mc_test == 0) | (y_mc_test == 1))

# Definir os dados para a run binária
x_bin, y_bin = x_mc[train_filter], y_mc[train_filter]
x_bin_test, y_bin_test = x_mc_test[test_filter], y_mc_test[test_filter]

# Definir run binária
runs.append([x_bin, y_bin, x_bin_test, y_bin_test])

for i, run in enumerate(runs):
    # Definição dos diretórios para os arquivos
    project = "multiclass" if i == 0 else "binary"
    run_dir = Path(f"runs/{project}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Definição do tuner para busca de hiperparâmetros
    tuner = keras_tuner.BayesianOptimization(build_model,
                                             objective='val_accuracy',
                                             max_trials=10,
                                             max_consecutive_failed_trials=5,
                                             directory=run_dir,
                                             project_name=f'{project}_model',
                                             )

    # Obter dados que serão utilizados
    x = run[0]
    y = run[1]
    x_test = run[2]
    y_test = run[3]

    # Separar os dados de treino entre treino e validação
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=42)

    # Redimensionar os dados para se adequar ao keras
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
    x_val = x_val.reshape(-1, 28, 28, 1).astype('float32')
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')

    # Normalizar os dados entre 0 e 1
    x_train /= 255.0
    x_val /= 255.0
    x_test /= 255.0

    # Categorizar as classes
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # Definir parada antecipada
    stop_early = tf.keras.callbacks.EarlyStopping(patience=5)

    # Procurar melhores hiperparâmetros
    tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val), verbose=1, callbacks=[stop_early])

    # Definir melhores hiperparâmetros
    best_hps = tuner.get_best_hyperparameters()[0]
    print(f"Melhores hiperparâmetros: {best_hps.values}")
    model = build_model(best_hps)

    # Retreinar o modelo para encontrar o número de épocas ideal
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), verbose=1, callbacks=[stop_early])

    # Definir época com mais acurácia
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Melhor época: %d' % (best_epoch,))

    # Guardar pesos iniciais
    initial_weights = [layer.get_weights() for layer in model.layers]

    # Retreinar o modelo considerando épocas e hiperparâmetros encontrados
    model = build_model(best_hps)
    x_all = np.concatenate((x_train, x_val))
    y_all = np.concatenate((y_train, y_val))
    model.fit(x=x_all, y=y_all, epochs=best_epoch)

    # Avaliar o modelo
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Guardar pesos finais
    final_weights = [layer.get_weights() for layer in model.layers]

    # Fazer predição do modelo
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Definir labels das classes utilizadas
    if project == "binary":
        classes = [0, 1]  # Apenas classes 0 e 1
    else:
        classes = range(10)  # Todas as classes do MNIST

    # Plotar a matriz de confusão
    # Calcular matriz de confusão apenas para as classes utilizadas
    conf_matrix_data = confusion_matrix(y_true_classes, y_pred_classes, labels=classes)
    conf_matrix_fig = plt.figure(figsize=(10, 8))
    sb.heatmap(conf_matrix_data, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {project}')
    conf_matrix_fig.savefig(run_dir / 'confusion_matrix.png')
    plt.close(conf_matrix_fig)

    # Plotar o gráfico de loss do treinamento
    loss_fig = plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    loss_fig.savefig(run_dir / 'loss_plot.png')
    plt.close(loss_fig)

    # Salvar resultados
    results = {
        "hyperparameters": best_hps.values,
        "training_history": history.history,
        "evaluation": {
            "test_loss": float(score[0]),
            "test_accuracy": float(score[1])
        },
        "best_epoch": best_epoch
    }

    hyperparameters = model.summary()
    print(hyperparameters)

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(run_dir / 'hyperparameters.pkl', 'wb') as f:
        pickle.dump(hyperparameters, f)

    with open(run_dir / 'initial_weights.pkl', 'wb') as f:
        pickle.dump(initial_weights, f)

    with open(run_dir / 'final_weights.pkl', 'wb') as f:
        pickle.dump(final_weights, f)

    with open(run_dir / 'training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    with open(run_dir / 'test_predictions.pkl', 'wb') as f:
        pickle.dump(y_pred, f)
