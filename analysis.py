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
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


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
    """Carrega os dados de otimização da CNN e gera gráficos de sensibilidade."""
    for exp_name in ["cnn_multiclass", "cnn_binary"]:
        path = base_dir / exp_name / "tuning_analysis.json"
        if not path.exists():
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame([{'val_accuracy': t['score'], **t['hyperparameters']} for t in data])
        if df.empty:
            continue

        task_title = "CNN - Tarefa Multiclasse" if "multiclass" in exp_name else "CNN - Tarefa Binária"
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'Análise de Sensibilidade a Hiperparâmetros\n({task_title})', fontsize=20)

        sns.scatterplot(data=df, x='lr', y='val_accuracy', ax=axes[0], hue='conv_layers', palette='coolwarm', s=100,
                        alpha=0.8)
        axes[0].set_xscale('log')
        axes[0].set_title('Taxa de Aprendizado vs. Acurácia', fontsize=16)
        axes[0].set_xlabel('Taxa de Aprendizado (Escala Log)')
        axes[0].set_ylabel('Acurácia de Validação do Trial')
        axes[0].grid(True, which="both", ls="--")
        axes[0].legend(title='Nº Camadas Conv.')

        sns.boxplot(data=df, x='conv_layers', y='val_accuracy', ax=axes[1], palette='crest')
        sns.stripplot(data=df, x='conv_layers', y='val_accuracy', ax=axes[1], color=".25", alpha=0.6)
        axes[1].set_title('Nº de Camadas de Convolução vs. Acurácia', fontsize=16)
        axes[1].set_xlabel('Número de Camadas de Convolução')
        axes[1].set_ylabel('')
        axes[1].grid(True, axis='y', ls="--")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_dir / f'analise_hps_{exp_name}.png')
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
        _plot_hyperparameter_analysis_for_cnn(base_dir, output_dir)
