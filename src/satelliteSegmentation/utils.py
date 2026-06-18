from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch


def plot_confusion_matrix(
    cm: np.ndarray, class_names: list | None = None, normalize: list | None = None
):
    cm = np.array(cm)

    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-7)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names if class_names else range(len(cm)),  # type: ignore
        yticklabels=class_names if class_names else range(len(cm)),  # type: ignore
    )

    plt.xlabel("Predicción")
    plt.ylabel("Ground Truth")
    plt.title("Matriz de Confusión (normalizada)")
    plt.show()

def plot_class_metrics(dice: list, iou: list, class_names=None):

    x = np.arange(len(dice))

    plt.figure(figsize=(10, 5))

    plt.plot(x, dice, marker="o", label="Dice")
    plt.plot(x, iou, marker="o", label="IoU")

    plt.xticks(
        x,
        class_names if class_names else [str(i) for i in x]
    )

    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Dice e IoU por clase")
    plt.xlabel("Clase")
    plt.ylabel("Score")

    plt.show()


def plot_bar_metrics(dice, iou):
    x = np.arange(len(dice))
    plt.figure(figsize=(10, 5))

    plt.bar(x - 0.2, dice, width=0.4, label="Dice")
    plt.bar(x + 0.2, iou, width=0.4, label="IoU")

    plt.ylim(0, 1)
    plt.xticks(x)
    plt.xlabel("Clase")
    plt.ylabel("Score")
    plt.title("Comparación Dice vs IoU por clase")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.show()