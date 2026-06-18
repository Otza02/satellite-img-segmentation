from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch

from satelliteSegmentation.tokenizer import IDX2LABEL


def plot_confusion_matrix(cm: np.ndarray):
    cm = np.array(cm)
    cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-7)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[IDX2LABEL[i] for i in range(len(IDX2LABEL))],
        yticklabels=[IDX2LABEL[i] for i in range(len(IDX2LABEL))],
    )

    plt.xlabel("Predicción")
    plt.ylabel("Ground Truth")
    plt.title("Matriz de Confusión (normalizada)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_class_metrics(dice: list, iou: list):
    class_names = [IDX2LABEL[i] for i in range(len(IDX2LABEL))]
    x = np.arange(len(class_names))
    plt.figure(figsize=(12, 5))

    plt.plot(x, dice, marker="o", label="Dice")
    plt.plot(x, iou, marker="o", label="IoU")

    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)

    plt.legend()
    plt.title("Dice e IoU por clase")
    plt.tight_layout()
    plt.show()


def plot_bar_metrics(dice, iou):
    class_names = [IDX2LABEL[i] for i in range(len(IDX2LABEL))]
    x = np.arange(len(class_names))
    plt.figure(figsize=(12, 5))

    plt.bar(x - 0.2, dice, width=0.4, label="Dice")
    plt.bar(x + 0.2, iou, width=0.4, label="IoU")

    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)

    plt.legend()
    plt.title("Dice vs IoU por clase")
    plt.tight_layout()
    plt.show()
