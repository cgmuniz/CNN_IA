# =====================================================================================
# ACH2016 - Inteligência Artificial (2025)
# EP2 - CNN
#
# 11796718 – LUCAS SARTOR CHAUVIN
# 12873188 – GABRIEL BERNARDINI SCHIMIDT
# 14564405 – CAUÃ GABRIEL MUNIZ DOS SANTOS
# 14592498 – LUIS YUDI ODAKE FERREIRA
# 14778136 – LEONEL MARCO ANTONIO MORGADO
# =====================================================================================

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
import json
from skimage.feature import hog
import pandas as pd
import time
from CNN_IA.analysis import run_full_analysis


# ---------------------------- Dados de Entrada ----------------------------
def load_and_preprocess_data(binary=False, test_size=0.33, feature_extractor='raw'):
    """
    Carrega o dataset MNIST, o pré-processa e o divide em conjuntos de treino, validação e teste.
    Retorna os dados como imagens brutas (para CNN) ou como vetores HOG (para MLP).
    Encapsula toda a lógica de preparação dos dados:
    1. Carrega os dados do Keras.
    2. Filtra para uma classificação binária (dígitos 0 e 1) se `binary=True`.
    3. Divide os dados de treino em conjuntos de treino e validação.
    4. Redimensiona as imagens (adiciona um canal de cor) e normaliza os pixels para o intervalo [0, 1].
    5. Converte os rótulos para o formato one-hot encoding.

    Argumentos:
        binary (bool): Filtra para classes 0 e 1 se True.
        test_size (float): Proporção para o conjunto de validação.
        feature_extractor (str): 'raw' para imagens brutas, 'hog' para características HOG

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

    if feature_extractor == 'hog':
        print("Extraindo características HOG...")
        x_train = extract_hog_features(x_train)
        x_val = extract_hog_features(x_val)
        x_test = extract_hog_features(x_test)
        print(f"Dimensão do vetor HOG: {x_train.shape[1]}")
    else:  # 'raw'
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

def extract_hog_features(images):
    """
    Extrai características HOG de um conjunto de imagens.

    Argumentos:
        images (np.array): Um array de imagens em escala de cinza (formato [n_samples, width, height]).

    Retorna:
        np.array: Um array de vetores de características HOG.
    """
    hog_features = []
    for image in images:
        # Extrai o vetor HOG para cada imagem.
        # Os parâmetros podem ser ajustados, mas estes são um bom ponto de partida para MNIST.
        features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False, transform_sqrt=True)
        hog_features.append(features)
    return np.array(hog_features)


def load_results(base_dir="runs"):
    """Carrega todos os arquivos results.json do diretório de execuções."""
    results = []
    run_paths = Path(base_dir).glob("**/results.json")

    for path in run_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            experiment_name = path.parent.name
            time_data = data.get("execution_time_seconds", {})
            results.append({
                "Experimento": experiment_name,
                "Modelo": "CNN" if "cnn" in experiment_name else "MLP+HOG",
                "Tarefa": "Binária" if "binary" in experiment_name else "Multiclasse",
                "Tempo de Tuning (s)": time_data.get("tuning_phase", 0),
                "Tempo de Treinamento (s)": time_data.get("final_training_phase", 0),
                "Tempo Total (s)": time_data.get("total", 0),
                "Acurácia": data.get("evaluation", {}).get("test_accuracy", 0)
            })
    return pd.DataFrame(results)

# ---------------------------- Modelo ----------------------------
def build_cnn_model(hp, num_classes=10):
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

def build_mlp_model(hp, input_shape, num_classes=10):
    """
    Constrói um modelo Multi-Layer Perceptron (MLP) com hiperparâmetros ajustáveis.

    Este modelo é ideal para dados tabulares ou vetores de características, como os
    extraídos pelo HOG. A arquitetura otimiza:
    - O número de camadas densas.
    - O número de neurônios em cada camada.
    - A presença e a taxa de dropout.
    - A taxa de aprendizado do otimizador Adam.

    Argumentos:
        hp (HyperParameters): Objeto do Keras Tuner para definir o espaço de busca.
        input_shape (int): A dimensionalidade do vetor de entrada (ex: tamanho do vetor HOG).
        num_classes (int): O número de neurônios na camada de saída.

    Retorna:
        keras.Model: O modelo MLP compilado.
    """
    model = keras.Sequential(name="MLP_Model")
    model.add(keras.Input(shape=(input_shape,)))
    for i in range(hp.Int('dense_layers', 1, 3)):
        model.add(layers.Dense(
            units=hp.Int(f'units_{i}', 64, 512, step=64),
            activation='relu'
        ))
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=hp.Float(f'dropout_{i}', 0.2, 0.5)))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
        loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

# ---------------------------- Treinamento e Otimização ----------------------------
def tune_model(model_builder, x_train, y_train, x_val, y_val, run_dir):
    """
    Executa a otimização de hiperparâmetros usando BayesianOptimization.

    Argumentos:
        model_builder (function): Uma função lambda que constrói o modelo.
        x_train, y_train: Dados de treinamento.
        x_val, y_val: Dados de validação.
        run_dir (Path): Diretório para salvar os resultados do tuner.

    Retorna:
        keras_tuner.Tuner: O objeto tuner após a busca, contendo todos os resultados.
    """
    # Definição do tuner para busca de hiperparâmetros
    tuner = keras_tuner.BayesianOptimization(
        hypermodel=model_builder,
        objective='val_accuracy',
        max_trials=2,  # todo: mudar aqui para entrega final, deixei baixo para testar
        directory=run_dir,
        project_name='tuning_trials'
    )

    # Começa timer para tempo de tuning
    start_time = time.time()

    # Busca os melhores hiperparâmetros, parando cedo se a performance não melhorar
    tuner.search(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=5,  # todo: mudar aqui para entrega final, deixei baixo para testar
        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    )

    # Calcula tempo de tuning
    end_time = time.time()
    tuning_time_seconds = end_time - start_time
    print(f"Tempo de tuning: {tuning_time_seconds:.2f} segundos")

    # Salvamos dados dos trials em um arquivo para análise posterior
    trials_data = []
    for trial in tuner.oracle.trials.values():
        if trial.status == "COMPLETED":
            trial_info = {
                'trial_id': trial.trial_id,
                'score': trial.score,
                'hyperparameters': trial.hyperparameters.values
            }
            trials_data.append(trial_info)
    with open(run_dir / "tuning_analysis.json", "w", encoding="utf-8") as f:
        json.dump(trials_data, f, indent=4)

    return tuner, tuning_time_seconds

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
    initial_weights = [layer.get_weights() for layer in model.layers]
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    start_time = time.time()
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=10, # todo: mudar aqui para entrega final, deixei baixo para testar
        callbacks=[early_stop]
    )

    end_time = time.time()
    training_time_seconds = end_time - start_time
    print(f"Tempo de treinamento final: {training_time_seconds:.2f} segundos")

    final_weights = [layer.get_weights() for layer in model.layers]

    return history, initial_weights, final_weights, training_time_seconds

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

def save_artifacts(run_dir, history, score, best_hps, y_pred_probs, initial_weights, final_weights, tuning_time, training_time):
    """
    Salva todos os artefatos do experimento: resultados, histórico, predições, pesos e tempos de execução.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # Função auxiliar para converter arrays NumPy em listas
    def convert_numpy_to_list(data):
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(data)
        if isinstance(data, (np.float64, np.float16, np.float32, np.float64)):
            return float(data)
        if isinstance(data, list):
            return [convert_numpy_to_list(item) for item in data]
        if isinstance(data, dict):
            return {key: convert_numpy_to_list(value) for key, value in data.items()}
        return data

    # Salva os resultados principais e tempos de execução
    results_data = {
        "hyperparameters": best_hps.values,
        "evaluation": {
            "test_loss": float(score[0]),
            "test_accuracy": float(score[1]),
        },
        "execution_time_seconds": {
            "tuning_phase": round(tuning_time, 2),
            "final_training_phase": round(training_time, 2),
            "total": round(tuning_time + training_time, 2),
        },
        "best_epoch": int(np.argmin(history.history['val_loss']) + 1)
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4)

    with open(run_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(convert_numpy_to_list(history.history), f, indent=4)

    artifacts_to_save = {
        "predictions.json": y_pred_probs,
        "initial_weights.json": initial_weights,
        "final_weights.json": final_weights,
    }

    for filename, data in artifacts_to_save.items():
        with open(run_dir / filename, "w", encoding="utf-8") as f:
            json.dump(convert_numpy_to_list(data), f)

# ---------------------------- Execução do Experimento ----------------------------
def run_experiment(task_name="binary", binary=True, model_type="cnn"):
    """
    Executa um fluxo completo de experimento de uma rede CNN ou MLP.

    Encapsula todas as etapas: configuração, carregamento de dados, otimização,
    treinamento, avaliação e salvamento de resultados e visualizações.

    Argumentos:
        task_name (str): Nome do experimento, usado para criar o diretório de saída.
        binary (bool): Define se o experimento é de classificação binária ou multiclasse.
        model_type (str): 'cnn' para usar imagens brutas ou 'mlp' para usar HOG.
    """
    print(f"--- Iniciando experimento: {task_name} ---")
    run_dir = Path("runs") / task_name

    # 1. Preparação dos dados
    feature_extractor = 'hog' if model_type == 'mlp' else 'raw'
    (x_train, y_train), (x_val, y_val), (x_test, y_test), num_classes = \
        load_and_preprocess_data(binary=binary, feature_extractor=feature_extractor)

    # 2. Definição do construtor do modelo
    if model_type == 'mlp':
        input_shape = x_train.shape[1]
        model_builder = lambda hp: build_mlp_model(hp, input_shape=input_shape, num_classes=num_classes)
    else:  # cnn
        model_builder = lambda hp: build_cnn_model(hp, num_classes=num_classes)

    # 3. Otimização de hiperparâmetros
    tuner, tuning_time = tune_model(
        model_builder=model_builder,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        run_dir=run_dir
    )

    # 4. Construção e Treinamento do modelo final
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Melhores hiperparâmetros encontrados para '{task_name}':")
    for param, value in best_hp.values.items():
        print(f" - {param}: {value}")

    model = tuner.hypermodel.build(best_hp)
    history, initial_weights, final_weights, training_time = train_final_model(
        model=model,
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val
    )

    # 5. Avaliação
    score, y_true, y_pred, y_pred_probs = evaluate(model=model, x_test=x_test, y_test=y_test)
    print(f"Resultado da Avaliação para '{task_name}': Loss={score[0]:.4f}, Accuracy={score[1]:.4f}\n")

    # 6. Salvamento dos artefatos e gráficos da execução
    save_artifacts(
        run_dir=run_dir, history=history, score=score, best_hps=best_hp,
        y_pred_probs=y_pred_probs, initial_weights=initial_weights, final_weights=final_weights,
        tuning_time=tuning_time, training_time=training_time
    )
    print(f"--- Experimento '{task_name}' concluído. Resultados salvos em: {run_dir} ---")

# ---------------------------- Main ----------------------------
if __name__ == "__main__":
    # Executa os experimentos CNN definidos: um multiclasse e um binário.
    run_experiment(task_name="cnn_multiclass", binary=False, model_type='cnn')
    run_experiment(task_name="cnn_binary", binary=True, model_type='cnn')

    # Executa os experimentos HOG definidos: um multiclasse e um binário.
    run_experiment(task_name="mlp_hog_multiclass", binary=False, model_type='mlp')
    run_experiment(task_name="mlp_hog_binary", binary=True, model_type='mlp')

    # Gera gráficos e imagens de análise
    run_full_analysis()
