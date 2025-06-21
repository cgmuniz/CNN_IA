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
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve


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

def _plot_conv_layer_time_analysis(df, output_dir):
    """
    Plota a relação entre a quantidade de camadas de convolução e o tempo de execução total,
    considerando apenas modelos do tipo CNN.
    """

    # Filtra apenas modelos CNN e remove dados inválidos
    df_cnn = df[df["Modelo"] == "CNN"].copy()
    df_cnn = df_cnn[["Tempo Total (s)", "Camadas de Convolucao", "Modelo", "Tarefa", "Experimento"]]
    df_cnn = df_cnn.dropna()
    df_cnn = df_cnn[np.isfinite(df_cnn["Tempo Total (s)"]) & np.isfinite(df_cnn["Camadas de Convolucao"])]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_cnn,
        x="Tempo Total (s)",
        y="Camadas de Convolucao",
        hue="Modelo",
        style="Tarefa",
        s=250,
        alpha=0.9
    )

    # Adiciona nome dos experimentos como rótulo nos pontos
    for i in range(df_cnn.shape[0]):
        plt.text(
            df_cnn["Tempo Total (s)"].iloc[i] + 1,
            df_cnn["Camadas de Convolucao"].iloc[i],
            str(df_cnn["Experimento"].iloc[i]),
            fontsize=9
        )

    plt.title("Trade-off: Camadas de Convolução vs. Tempo de Execução", fontsize=18, pad=20)
    plt.xlabel("Tempo Total de Execução (s)", fontsize=12)
    plt.ylabel("Camadas de Convolucao", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Legenda")
    plt.tight_layout()
    plt.savefig(output_dir / "comparacao_camadas_conv_vs_tempo.png")
    plt.close()

def _plot_lr_vs_time(df, output_dir):
    """
    Plota a relação entre taxa de aprendizado e tempo de execução total.
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df, x="Tempo Total (s)", y="Taxa de Aprendizado",
        hue="Modelo", style="Tarefa", s=250, alpha=0.9
    )
    for i in range(df.shape[0]):
        plt.text(df['Tempo Total (s)'][i] + 5, df['Taxa de Aprendizado'][i], df['Experimento'][i], fontsize=9)
    plt.title("Trade-off: Taxa de Aprendizado vs. Tempo de Execução", fontsize=18, pad=20)
    plt.xlabel("Tempo Total de Execução (s)", fontsize=12)
    plt.ylabel("Taxa de Aprendizado", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Legenda")
    plt.tight_layout()
    plt.savefig(output_dir / "comparacao_lr_vs_tempo.png")
    plt.close()

def _load_all_results_df(base_dir="runs"):
    """Função interna para carregar os resultados finais de todos os experimentos."""
    results_list = []
    for path in Path(base_dir).glob("**/results.json"):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        exp_name = path.parent.name
        time_data = data.get("execution_time_seconds", {})
        hyperparameters_data = data.get("hyperparameters", {})
        results_list.append({
            "Experimento": exp_name,
            "Modelo": "CNN" if "cnn" in exp_name else "MLP+HOG",
            "Tarefa": "Binária" if "binary" in exp_name else "Multiclasse",
            "Tempo Total (s)": time_data.get("total", 0),
            "Acurácia": data.get("evaluation", {}).get("test_accuracy", 0),
            "Camadas de Convolucao": hyperparameters_data.get("conv_layers"),
            "Taxa de Aprendizado": hyperparameters_data.get("lr")
        })
    return pd.DataFrame(results_list)

def _plot_roc_curve(run_dir):
    """
    Plota e salva o gráfico da curva ROC para classificação binária.

    Parâmetros:
        run_dir (Path ou str): Caminho para o diretório do experimento (ex: 'runs/cnn_binary')
    """
    run_dir = Path(run_dir)

    # Carrega rótulos verdadeiros
    with open(run_dir / "evaluation_outputs.json", "r") as f:
        eval_data = json.load(f)
    y_true = np.array(eval_data["y_true"])

    # Carrega probabilidades previstas
    with open(run_dir / "predictions.json", "r") as f:
        y_pred_probs = np.array(json.load(f))
    y_positive_probs = y_pred_probs[:, 1]  # probabilidade da classe 1

    # Calcula FPR, TPR e AUC
    fpr, tpr, _ = roc_curve(y_true, y_positive_probs)
    roc_auc = auc(fpr, tpr)

    # Plota a curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title(f'Curva ROC - {run_dir.name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    # Salva imagem
    plt.savefig(run_dir / "roc_curve.png")
    plt.close()
    print(f"Gráfico ROC salvo em: {run_dir / 'roc_curve.png'}")

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
        pred_path = run_dir / "predictions.json"

        if history_path.exists():
            with open(history_path, 'r') as f:
                plot_learning_curves(run_dir, json.load(f))

        if eval_path.exists():
            with open(eval_path, 'r') as f:
                eval_data = json.load(f)
            num_classes = 10 if "multiclass" in run_dir.name else 2
            plot_confusion_matrix(run_dir, eval_data, num_classes)

        if pred_path.exists() and "binary" in run_dir.name:
            _plot_roc_curve(run_dir)

    # 2. Gerar gráficos comparativos finais
    results_df = _load_all_results_df(base_dir)
    if not results_df.empty:
        output_dir = Path('.')
        _plot_accuracy_time_tradeoff(results_df, output_dir)
        _plot_hyperparameter_analysis_for_cnn(base_dir, output_dir)
        _plot_conv_layer_time_analysis(results_df, output_dir)
        _plot_lr_vs_time(results_df, output_dir)
    