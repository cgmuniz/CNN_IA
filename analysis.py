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
import keras_tuner
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import confusion_matrix
from models import build_cnn_model


def plot_learning_curves(run_dir, history_data):
    """Gera um gráfico com as curvas de aprendizado de loss e acurácia."""
    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Curvas de Aprendizado - {run_dir.name}", fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(history_data['loss'], label='Loss de Treino')
    plt.plot(history_data['val_loss'], label='Loss de Validação')
    plt.title('Loss vs. Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_data['accuracy'], label='Acurácia de Treino')
    plt.plot(history_data['val_accuracy'], label='Acurácia de Validação')
    plt.title('Acurácia vs. Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(run_dir / 'learning_curves.png')
    plt.close()


def plot_confusion_matrix(run_dir, eval_data, num_classes):
    """Gera a matriz de confusão."""
    y_true, y_pred = eval_data['y_true'], eval_data['y_pred']
    labels = range(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Matriz de Confusão - {run_dir.name}', fontsize=16)
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.savefig(run_dir / 'confusion_matrix.png')
    plt.close()


def _load_all_results_df(base_dir="runs"):
    """Função interna para carregar os resultados finais de todos os experimentos."""
    results_list = []
    for path in Path(base_dir).glob("**/results.json"):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        exp_name = path.parent.name
        time_data = data.get("execution_time_seconds", {})
        results_list.append({
            "Experimento": exp_name,
            "Modelo": "CNN" if "cnn" in exp_name else "MLP+HOG",
            "Tarefa": "Binária" if "binary" in exp_name else "Multiclasse",
            "Tempo Total (s)": time_data.get("total", 0),
            "Acurácia": data.get("evaluation", {}).get("test_accuracy", 0)
        })
    return pd.DataFrame(results_list)


def _plot_accuracy_time_tradeoff(df, output_dir):
    """Cria um gráfico de dispersão para analisar o trade-off entre acurácia e tempo."""
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df, x="Tempo Total (s)", y="Acurácia",
        hue="Modelo", style="Tarefa", s=250, alpha=0.9
    )
    for i in range(df.shape[0]):
        plt.text(df['Tempo Total (s)'][i] + 5, df['Acurácia'][i], df['Experimento'][i], fontsize=9)
    plt.title("Trade-off: Acurácia vs. Tempo de Execução", fontsize=18, pad=20)
    plt.xlabel("Tempo Total de Execução (s)", fontsize=12)
    plt.ylabel("Acurácia no Teste", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Legenda")
    plt.tight_layout()
    plt.savefig(output_dir / "comparacao_acuracia_vs_tempo.png")
    plt.close()


def _plot_hyperparameter_analysis_for_cnn(base_dir, output_dir):
    """
    Carrega os dados de otimização da CNN e gera uma análise visual detalhada
    do impacto de múltiplos hiperparâmetros no desempenho.
    """
    # Itera sobre os dois experimentos da CNN
    for exp_name in ["cnn_multiclass", "cnn_binary"]:
        path = base_dir / exp_name / "tuning_analysis.json"
        if not path.exists():
            continue

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Transforma os dados em um DataFrame do Pandas para fácil manipulação
        records = [{'val_accuracy': t['score'], **t['hyperparameters']} for t in data]
        df = pd.DataFrame(records)
        if df.empty:
            continue

        # --- Geração dos Gráficos ---
        task_title = "CNN - Tarefa Multiclasse" if "multiclass" in exp_name else "CNN - Tarefa Binária"

        # Cria uma figura com 4 subplots (2x2) para uma análise completa
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'Análise de Sensibilidade a Hiperparâmetros\n({task_title})', fontsize=22)

        # Gráfico 1: Taxa de Aprendizado (lr)
        ax = axes[0, 0]
        sns.scatterplot(data=df, x='lr', y='val_accuracy', hue='conv_layers', palette='coolwarm', s=100, alpha=0.8,
                        ax=ax)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())  # Formata os rótulos para decimal
        ax.set_title('Taxa de Aprendizado vs. Acurácia', fontsize=16)
        ax.set_xlabel('Taxa de Aprendizado', fontsize=12)
        ax.set_ylabel('Acurácia de Validação do Trial', fontsize=12)
        ax.grid(True, which="both", ls="--")
        ax.legend(title='Nº Camadas Conv.')

        # Gráfico 2: Número de Filtros na 1ª Camada (filters_0)
        ax = axes[0, 1]
        sns.boxplot(data=df, x='filters_0', y='val_accuracy', ax=ax, palette='magma')
        sns.stripplot(data=df, x='filters_0', y='val_accuracy', ax=ax, color=".25", alpha=0.6)
        ax.set_title('Nº de Filtros (1ª Camada) vs. Acurácia', fontsize=16)
        ax.set_xlabel('Número de Filtros', fontsize=12)
        ax.set_ylabel('')  # Remove rótulo Y para um visual mais limpo
        ax.grid(True, axis='y', ls="--")

        # Gráfico 3: Tamanho do Kernel na 1ª Camada (kernel_size_0)
        ax = axes[1, 0]
        # Para o kernel, o boxplot é ideal para comparar as duas categorias (1 e 5)
        sns.boxplot(data=df, x='kernel_size_0', y='val_accuracy', ax=ax, palette='viridis')
        sns.stripplot(data=df, x='kernel_size_0', y='val_accuracy', ax=ax, color=".25", alpha=0.6)
        ax.set_title('Tamanho do Kernel (1ª Camada) vs. Acurácia', fontsize=16)
        ax.set_xlabel('Tamanho do Kernel', fontsize=12)
        ax.set_ylabel('Acurácia de Validação do Trial', fontsize=12)
        ax.grid(True, axis='y', ls="--")

        # Gráfico 4: Tamanho do Pooling na 1ª Camada (pool_size_0)
        ax = axes[1, 1]
        # Verifica se o hiperparâmetro existe, pois ele pode não estar em todos os trials
        if 'pool_size_0' in df.columns:
            sns.boxplot(data=df, x='pool_size_0', y='val_accuracy', ax=ax, palette='plasma')
            sns.stripplot(data=df, x='pool_size_0', y='val_accuracy', ax=ax, color=".25", alpha=0.6)
            ax.set_title('Tamanho do Pooling (1ª Camada) vs. Acurácia', fontsize=16)
            ax.set_xlabel('Tamanho do Pooling (2x2 ou 3x3)', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'Dados de "pool_size_0"\nnão encontrados nos trials.',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, transform=ax.transAxes)
            ax.set_title('Tamanho do Pooling (1ª Camada)', fontsize=16)
        ax.set_ylabel('')
        ax.grid(True, axis='y', ls="--")

        # Ajusta o layout geral para evitar sobreposição e salva a figura
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_filename = f'analise_hps_detalhada_{exp_name}.png'
        plt.savefig(output_dir / output_filename)
        plt.close()


def _plot_pareto_front_analysis(base_dir, output_dir):
    """
    Analisa todos os trials da CNN para visualizar o trade-off entre
    performance (acurácia) e complexidade (número de parâmetros).
    """
    print("\n--- Gerando Análise de Eficiência do Modelo (Fronteira de Pareto) ---")
    for exp_name in ["cnn_multiclass", "cnn_binary"]:
        path = base_dir / exp_name / "tuning_analysis.json"
        if not path.exists():
            continue

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Para cada trial, vamos reconstruir o modelo para contar seus parâmetros
        trial_analysis = []
        for trial in data:
            hp = keras_tuner.HyperParameters()
            # Precisamos recarregar os HPs do trial no objeto hp
            for key, value in trial['hyperparameters'].items():
                hp.Fixed(key, value=value)

            # Reconstruímos o modelo apenas para contar os parâmetros
            num_classes = 10 if "multiclass" in exp_name else 2
            model = build_cnn_model(hp, num_classes)

            trial_analysis.append({
                'val_accuracy': trial['score'],
                'parameters': model.count_params(),
                'trial_id': trial['trial_id']
            })

        df = pd.DataFrame(trial_analysis)
        if df.empty:
            continue

        # Encontra o melhor trial para destacá-lo no gráfico
        best_trial_idx = df['val_accuracy'].idxmax()
        best_trial = df.loc[best_trial_idx]

        # Geração do Gráfico
        task_title = "CNN - Tarefa Multiclasse" if "multiclass" in exp_name else "CNN - Tarefa Binária"
        plt.figure(figsize=(14, 8))

        ax = sns.scatterplot(
            data=df,
            x='parameters',
            y='val_accuracy',
            palette='viridis',
            s=80,
            alpha=0.7
        )

        # Destaca o melhor ponto
        ax.scatter(
            best_trial['parameters'], best_trial['val_accuracy'],
            color='red', s=200, edgecolor='black', zorder=5,
            label=f"Melhor Trial ({best_trial['val_accuracy']:.4f})"
        )

        ax.set_title(f'Análise de Eficiência: Acurácia vs. Complexidade do Modelo\n({task_title})', fontsize=18, pad=20)
        ax.set_xlabel('Número de Parâmetros Treináveis (Complexidade)', fontsize=12)
        ax.set_ylabel('Acurácia de Validação do Trial', fontsize=12)
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()

        # Formata o eixo X para ser mais legível (ex: 1.5M em vez de 1500000)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x / 1_000_000:.1f}M'))

        plt.tight_layout()
        output_filename = f'analise_eficiencia_{exp_name}.png'
        plt.savefig(output_dir / output_filename)
        plt.close()

def run_full_analysis():
    """
    Função pública que orquestra toda a geração de gráficos de análise.
    Esta é a única função que o main.py precisará chamar.
    """
    base_dir = Path("runs")
    if not base_dir.exists() or not any(base_dir.iterdir()):
        print("Pasta 'runs' não encontrada ou vazia. Execute o 'main.py' para gerar os dados primeiro.")
        return

    # 1. Gerar gráficos individuais para cada experimento
    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue

        history_path = run_dir / "training_history.json"
        eval_path = run_dir / "evaluation_outputs.json"

        if history_path.exists():
            with open(history_path, 'r') as f:
                plot_learning_curves(run_dir, json.load(f))

        if eval_path.exists():
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            num_classes = 10 if "multiclass" in run_dir.name else 2
            plot_confusion_matrix(run_dir, eval_data, num_classes)

    # 2. Gerar gráficos comparativos finais
    results_df = _load_all_results_df(base_dir)
    if not results_df.empty:
        output_dir = Path('.')
        _plot_accuracy_time_tradeoff(results_df, output_dir)
       # _plot_hyperparameter_analysis_for_cnn(base_dir, output_dir)
       # _plot_pareto_front_analysis(base_dir, output_dir)


def main():
    run_full_analysis()
