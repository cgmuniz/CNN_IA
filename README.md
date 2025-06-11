# 🤖 EP2 - Redes Neurais CNN e MLP com HOG para Classificação de Dígitos (MNIST)

**Disciplina:** ACH2016 - Inteligência Artificial (2025)  
**Projeto:** Exercício-Programa 2

**Autores:**
*   11796718 – LUCAS SARTOR CHAUVIN
*   12873188 – GABRIEL BERNARDINI SCHIMIDT
*   14564405 – CAUÃ GABRIEL MUNIZ DOS SANTOS
*   14592498 – LUIS YUDI ODAKE FERREIRA
*   14778136 – LEONEL MARCO ANTONIO MORGADO

---

## 📜 Visão Geral do Projeto

Este projeto implementa e compara duas abordagens distintas para o problema de classificação de dígitos do dataset MNIST:

1.  **Rede Neural Convolucional (CNN):** Um modelo de deep learning que aprende a extrair características automaticamente a partir dos pixels brutos da imagem.
2.  **Multi-Layer Perceptron (MLP) com HOG:** Uma abordagem mais tradicional onde as características são extraídas manualmente usando o descritor HOG (Histogram of Oriented Gradients) e, em seguida, utilizadas para treinar uma rede neural densa (MLP).

O código executa quatro experimentos distintos para uma análise completa:
*   CNN em tarefa binária (dígitos 0 vs. 1)
*   CNN em tarefa multiclasse (dígitos 0 a 9)
*   MLP + HOG em tarefa binária
*   MLP + HOG em tarefa multiclasse

Para cada experimento, o projeto utiliza o **Keras Tuner** para otimizar os hiperparâmetros do modelo, garantindo uma comparação justa e robusta entre as arquiteturas.

## ✨ Principais Características

*   **Otimização de Hiperparâmetros:** Utiliza `keras_tuner` com Otimização Bayesiana para encontrar as melhores arquiteturas de modelo.
*   **Comparação de Abordagens:** Avalia de forma sistemática o desempenho da extração de características aprendida (CNN) vs. manual (HOG).
*   **Geração Automática de Artefatos:** Salva automaticamente os resultados, métricas, gráficos e pesos de cada modelo para análise posterior.
*   **Visualização Comparativa:** Gera uma tabela e um gráfico de barras ao final da execução, comparando a acurácia de todos os experimentos.

## 🚀 Como Executar

Para executar o projeto, siga os passos abaixo.

### 1. Pré-requisitos

*   Python 3.9 ou superior

### 2. Instalação

**a. Crie e ative um ambiente virtual (altamente recomendado):**
```bash
# Comando para criar o ambiente virtual
python -m venv venv

# Ativar no Windows
.\venv\Scripts\activate

# Ativar no macOS/Linux
source venv/bin/activate
```

**b. Instale as dependências:**
O arquivo `requirements.txt` contém todas as bibliotecas necessárias. Instale-as com um único comando:
```bash
pip install -r requirements.txt
```

### 3. Execução do Script Principal

Execute o arquivo `main.py`. O script irá rodar todos os quatro experimentos em sequência.

```bash
python main.py
```
**Atenção:** A execução pode demorar um pouco, pois envolve otimização e treinamento de quatro modelos neurais. Ao final, os resultados e gráficos serão gerados.

## 📊 Artefatos Gerados

Após a execução, os seguintes artefatos serão criados na pasta do projeto:

**1. Estrutura de Pastas `runs/`:**
Uma pasta `runs` será criada, contendo subdiretórios para cada um dos quatro experimentos:

```
runs/
├── cnn_binary/
├── cnn_multiclass/
├── mlp_hog_binary/
└── mlp_hog_multiclass/
```

**2. Conteúdo de cada pasta de experimento:**
Dentro de cada subdiretório (ex: `runs/cnn_multiclass/`), você encontrará:

*   `results.json`: Um arquivo JSON com os melhores hiperparâmetros encontrados e as métricas finais de avaliação (loss e acurácia).
*   `confusion_matrix.png`: Gráfico da matriz de confusão do modelo no conjunto de teste.
*   `learning_curve.png`: Gráfico da curva de aprendizado (Loss vs. Épocas).
*   `accuracy_curve.png`: Gráfico da curva de acurácia (Acurácia vs. Épocas).
*   `history.pkl`: Arquivo Pickle com o histórico completo de treinamento.
*   `initial_weights.pkl` / `final_weights.pkl`: Pesos do modelo antes e depois do treinamento.
*   `predictions.pkl`: Predições do modelo no conjunto de teste.
*   `tuning_trials/`: Pasta gerada pelo Keras Tuner com os detalhes de cada tentativa de otimização.

**3. Gráfico Comparativo Final:**
Na raiz do projeto, um arquivo de imagem será gerado:

*   `comparison_accuracy.png`: Um gráfico de barras comparando a acurácia final de todos os quatro experimentos, facilitando a análise dos resultados.