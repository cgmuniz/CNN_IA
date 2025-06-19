from tensorflow import keras
from tensorflow.keras import layers


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
    for i in range(hp.Int(name='conv_layers', min_value=1, max_value=3)):
        model.add(layers.Conv2D(
            filters=hp.Int(name=f'filters_{i}', min_value=32, max_value=128, step=32),
            kernel_size=hp.Choice(name=f'kernel_size_{i}', values=[1, 5]),
            activation=hp.Choice(name=f'conv_activation_{i}', values=['relu', 'tanh'])
        ))
        model.add(layers.MaxPooling2D(pool_size=hp.Choice(name=f'pool_size_{i}', values=[2, 3])))

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