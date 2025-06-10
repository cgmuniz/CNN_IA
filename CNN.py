# -*- coding: utf-8 -*-
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import keras_tuner
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pathlib import Path
import pickle
import json

# ---------------------------- Dados de Entrada ----------------------------
def load_and_preprocess_data(binary=False, test_size=0.33):
    """
    Carrega o dataset MNIST, o pré-processa e o divide em conjuntos de treino, validação e teste.

    Esta função encapsula toda a lógica de preparação dos dados:
    1. Carrega os dados do Keras.
    2. Filtra para uma classificação binária (dígitos 0 e 1) se `binary=True`.
    3. Divide os dados de treino em conjuntos de treino e validação.
    4. Redimensiona as imagens (adiciona um canal de cor) e normaliza os pixels para o intervalo [0, 1].
    5. Converte os rótulos para o formato one-hot encoding.

    Argumentos:
        binary (bool): Se True, filtra os dados para usar apenas as classes 0 e 1.
        test_size (float): A proporção do dataset a ser alocada para o conjunto de validação.

    Retorna:
        tuple: Uma tupla contendo os conjuntos de dados ((x_train, y_train), (x_val, y_val), (x_test, y_test))
               e o número de classes.
    """
    (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

    if binary:
        # Filtra os dados para manter apenas as classes 0 e 1 para um problema binário
        train_filter = np.where((y == 0) | (y == 1))
        test_filter = np.where((y_test == 0) | (y_test == 1))
        x, y = x[train_filter], y[train_filter]
        x_test, y_test = x_test[test_filter], y_test[test_filter]
        num_classes = 2
    else:
        num_classes = 10

    # Divide os dados originais de treino em novos conjuntos de treino e validação
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    # Função auxiliar para redimensionar e normalizar imagens
    def reshape_and_normalize(data):
        return data.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    x_train = reshape_and_normalize(x_train)
    x_val = reshape_and_normalize(x_val)
    x_test = reshape_and_normalize(x_test)

    # Converte os rótulos de inteiros para vetores categóricos (one-hot encoding)
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes=num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), num_classes


# ---------------------------- Modelo ----------------------------
def build_model(hp, num_classes=10):
    """
    Constrói o modelo CNN com hiperparâmetros ajustáveis pelo Keras Tuner.

    A arquitetura é dinâmica, permitindo que o Keras Tuner otimize:
    - O número de camadas convolucionais e densas.
    - O número de filtros e unidades em cada camada.
    - O tamanho do kernel e a função de ativação.
    - A presença e a taxa de dropout.
    - A taxa de aprendizado do otimizador Adam.

    Argumentos:
        hp (HyperParameters): Objeto do Keras Tuner para definir o espaço de busca.
        num_classes (int): O número de neurônios na camada de saída (10 para multiclasse, 2 para binário).

    Retorna:
        keras.Model: O modelo Keras compilado.
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(28, 28, 1)))

    # Constrói blocos convolucionais dinamicamente
    for i in range(hp.Int(name='conv_layers', min_value=1, max_value=2)):
        model.add(layers.Conv2D(
            filters=hp.Int(name=f'filters_{i}', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice(name=f'kernel_size_{i}', values=[3, 5]),
            activation=hp.Choice(name=f'conv_activation_{i}', values=['relu', 'tanh'])
        ))
        model.add(layers.MaxPooling2D(pool_size=2))

    model.add(layers.Flatten())

    # Constrói camadas densas dinamicamente
    for i in range(hp.Int(name='dense_layers', min_value=1, max_value=2)):
        model.add(layers.Dense(
            units=hp.Int(name=f'units_{i}', min_value=64, max_value=256, step=64),
            activation=hp.Choice(name=f'dense_activation_{i}', values=['relu', 'tanh'])
        ))
        if hp.Boolean(name='dropout'):
            model.add(layers.Dropout(rate=hp.Float(name='dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))

    # Camada de saída com ativação softmax para classificação
    model.add(layers.Dense(units=num_classes, activation='softmax'))

    # Define taxa de aprendizado dinamicamente
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    # Compila o modelo com otimizador e função de perda
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ---------------------------- Treinamento e Otimização ----------------------------
def tune_model(x_train, y_train, x_val, y_val, run_dir, num_classes):
    """
    Executa a otimização de hiperparâmetros usando BayesianOptimization.

    Argumentos:
        x_train, y_train: Dados de treinamento.
        x_val, y_val: Dados de validação.
        run_dir (Path): Diretório para salvar os resultados do tuner.
        num_classes (int): Número de classes para passar para a função `build_model`.

    Retorna:
        keras_tuner.Tuner: O objeto tuner após a busca.
    """

    # Definição do tuner para busca de hiperparâmetros
    tuner = keras_tuner.BayesianOptimization(
        hypermodel=lambda hp: build_model(hp, num_classes),
        objective='val_accuracy',
        max_trials=10,
        directory=run_dir,
        project_name='tuning_trials'
    )

    # Busca os melhores hiperparâmetros, parando cedo se a performance não melhorar
    tuner.search(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    )

    return tuner

def train_final_model(model, x_train, y_train, x_val, y_val):
    """
    Treina o modelo final com os melhores hiperparâmetros encontrados.

    Argumentos:
        model (keras.Model): O modelo a ser treinado.
        x_train, y_train: Dados de treinamento.
        x_val, y_val: Dados de validação.

    Retorna:
        History: Objeto retornado pelo `model.fit` contendo o histórico de treinamento.
    """
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        callbacks=[early_stop]
    )
    return history

# ----------------------------  Avaliação e Visualização ----------------------------
def evaluate(model, x_test, y_test):
    """
    Avalia o modelo treinado no conjunto de teste.

    Argumentos:
        model (keras.Model): O modelo treinado.
        x_test, y_test: Dados de teste.

    Retorna:
        tuple: Contém a pontuação (loss, accuracy), rótulos verdadeiros, rótulos preditos e
               as probabilidades das predições.
    """
    score = model.evaluate(x=x_test, y=y_test, verbose=0)
    y_pred_probs = model.predict(x=x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    return score, y_true, y_pred, y_pred_probs

def save_artifacts(run_dir, history, score, best_hps, y_pred_probs, initial_weights, final_weights):
    """
    Salva todos os artefatos do experimento: resultados, histórico, predições e pesos.

    Argumentos:
        run_dir (Path): Diretório para salvar os artefatos.
        history (History): Histórico de treinamento.
        score (list): Lista com loss e accuracy do teste.
        best_hps (HyperParameters): Melhores hiperparâmetros encontrados.
        y_pred_probs (np.array): Predições (probabilidades) no conjunto de teste.
        initial_weights (list): Pesos do modelo antes do treinamento.
        final_weights (list): Pesos do modelo após o treinamento.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # Salva os resultados principais em um arquivo JSON
    results_data = {
        "hyperparameters": best_hps.values,
        "evaluation": {"test_loss": float(score[0]), "test_accuracy": float(score[1])},
        "best_epoch": len(history.history["val_accuracy"])
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(obj=results_data, fp=f, indent=2)

    # Salva outros artefatos usando pickle
    artifacts = {
        "history.pkl": history.history,
        "predictions.pkl": y_pred_probs,
        "initial_weights.pkl": initial_weights,
        "final_weights.pkl": final_weights
    }
    for fname, data in artifacts.items():
        with open(run_dir / fname, "wb") as f:
            pickle.dump(obj=data, file=f)

def plot_confusion_matrix(run_dir, y_true, y_pred, classes):
    """Gera e salva a imagem da matriz de confusão."""
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(data=cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.savefig(fname=run_dir / 'confusion_matrix.png')
    plt.close()

def plot_learning_curve(run_dir, history):
    """Gera e salva o gráfico da curva de aprendizado (loss)."""
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Loss de Treino')
    plt.plot(history.history['val_loss'], label='Loss de Validação')
    plt.title('Curva de Aprendizado')
    plt.xlabel('Épocas')
    plt.ylabel('Loss (Perda)')
    plt.legend()
    plt.grid(True)
    plt.savefig(fname=run_dir / 'learning_curve.png')
    plt.close()

def plot_accuracy_curve(run_dir, history):
    """Gera e salva o gráfico da curva de acurácia (accuracy)."""
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Acurácia de Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Curva de Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    plt.savefig(fname=run_dir / 'accuracy_curve.png')
    plt.close()

# ---------------------------- Execução do Experimento ----------------------------
def run_experiment(task_name="binary", binary=True):
    """
    Executa um fluxo completo de experimento de uma rede CNN.

    Encapsula todas as etapas: configuração, carregamento de dados, otimização,
    treinamento, avaliação e salvamento de resultados e visualizações.

    Argumentos:
        task_name (str): Nome do experimento, usado para criar o diretório de saída.
        binary (bool): Define se o experimento é de classificação binária ou multiclasse.
    """
    print(f"--- Iniciando experimento: {task_name} ---")
    run_dir = Path("runs") / task_name

    # 1. Preparação dos dados
    (x_train, y_train), (x_val, y_val), (x_test, y_test), num_classes = load_and_preprocess_data(binary=binary)

    # 2. Otimização de hiperparâmetros
    tuner = tune_model(x_train, y_train, x_val, y_val, run_dir=run_dir, num_classes=num_classes)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"Melhores hiperparâmetros encontrados para '{task_name}':")
    for param, value in best_hp.values.items():
        print(f" - {param}: {value}")

    # 3. Treinamento do modelo final
    model = build_model(hp=best_hp, num_classes=num_classes)
    initial_weights = [layer.get_weights() for layer in model.layers]
    history = train_final_model(model=model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    final_weights = [layer.get_weights() for layer in model.layers]

    # 4. Avaliação
    score, y_true, y_pred, y_pred_probs = evaluate(model=model, x_test=x_test, y_test=y_test)
    print(f"Resultado da Avaliação para '{task_name}': Loss={score[0]:.4f}, Accuracy={score[1]:.4f}\n")

    # 5. Salvamento dos artefatos e visualizações
    save_artifacts(
        run_dir=run_dir, history=history, score=score, best_hps=best_hp,
        y_pred_probs=y_pred_probs, initial_weights=initial_weights, final_weights=final_weights
    )
    plot_confusion_matrix(run_dir=run_dir, y_true=y_true, y_pred=y_pred, classes=range(num_classes))
    plot_learning_curve(run_dir=run_dir, history=history)
    plot_accuracy_curve(run_dir=run_dir, history=history)

    print(f"--- Experimento '{task_name}' concluído. Resultados salvos em: {run_dir} ---")

# ---------------------------- Main ----------------------------
if __name__ == "__main__":
    # Executa os dois experimentos definidos: um multiclasse e um binário.
    run_experiment(task_name="multiclass", binary=False)
    run_experiment(task_name="binary", binary=True)
