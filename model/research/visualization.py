import numpy as np
import pandas as pd
import seaborn as sn
import torch
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve


def print_binary_confusion_matrix(y_true, y_pred, output_path=None, title_suffix=""):
    cm = confusion_matrix(y_true, y_pred)
    print()
    df_cm = pd.DataFrame(cm, range(2), range(2))
    plt.clf()
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 16})
    plt.ylabel("Actual label", size=20)
    plt.xlabel("Predicted label", size=20)
    plt.xticks(np.arange(2), ["Fake", "Real"], size=16)
    plt.yticks(np.arange(2), ["Fake", "Real"], size=16)
    plt.ylim([2, 0])
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)

    calculated_acc = (cm[0][0] + cm[1][1]) / np.sum(cm)
    print("Calculated Accuracy", calculated_acc * 100)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))


def print_method_confusion_matrix(y_true_method, y_pred_method, output_path=None):
    labels = ["original", "Deepfakes", "FaceShifter", "FaceSwap", "NeuralTextures", "Face2Face", "others"]
    label_indices = list(range(len(labels)))
    cm = confusion_matrix(y_true_method, y_pred_method, labels=label_indices)
    print()
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.clf()
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 12}, cmap="Blues")
    plt.ylabel("Actual label", size=16)
    plt.xlabel("Predicted label", size=16)
    plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45, ha="right", fontsize=12)
    plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0, fontsize=12)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)

    print(f"Calculated Accuracy: {np.trace(cm) / np.sum(cm) * 100:.2f}%")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:")
    print(classification_report(y_true_method, y_pred_method, target_names=labels, labels=label_indices))


def plot_roc_curve(true_bin, output_bin, output_path=None, title_suffix=""):
    pred_score = torch.softmax(output_bin, dim=1)[:, 1].cpu().numpy()
    fpr, tpr, _ = roc_curve(true_bin, pred_score)
    roc_auc = auc(fpr, tpr)
    print(f"ROC curve (AUC = {roc_auc:.2f})")

    plt.clf()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Binary Classification){title_suffix}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)


def plot_tsne(features, labels, output_path=None, method_labels=None, random_state=42):
    n_samples = features.shape[0]
    perplexity = min(30, max(5, n_samples // 3))
    embedded = TSNE(n_components=2, perplexity=perplexity, random_state=random_state).fit_transform(features)
    labels = np.array(labels)

    plt.clf()
    plt.figure(figsize=(8, 6))
    for cls in np.unique(labels):
        idx = labels == cls
        label_name = method_labels[cls] if method_labels and cls < len(method_labels) else str(cls)
        plt.scatter(embedded[idx, 0], embedded[idx, 1], label=label_name, alpha=0.7)

    plt.legend()
    plt.title("t-SNE of Method Class Features")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)


def _checkpoint_output_path(checkpoint_path, checkpoint_name, suffix):
    return Path(checkpoint_path) / f"{checkpoint_name}_{suffix}.png"


def save_loss_curve(train_loss_avg, test_loss_avg, num_epochs, checkpoint_path, checkpoint_name):
    loss_train = train_loss_avg
    loss_val = test_loss_avg
    print(num_epochs)
    epochs = range(1, num_epochs + 1)
    plt.clf()
    plt.plot(epochs, loss_train, "g", label="Training loss")
    plt.plot(epochs, loss_val, "b", label="validation loss")
    plt.title("Training and Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(Path(checkpoint_path) / f"{checkpoint_name}_loss_plot.png")


def save_accuracy_curve(train_accuracy, test_accuracy, num_epochs, checkpoint_path, checkpoint_name):
    loss_train = train_accuracy
    loss_val = test_accuracy
    epochs = range(1, num_epochs + 1)
    plt.clf()
    plt.plot(epochs, loss_train, "g", label="Training accuracy")
    plt.plot(epochs, loss_val, "b", label="validation accuracy")
    plt.title("Training and Validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(Path(checkpoint_path) / f"{checkpoint_name}_accuracy_plot.png")


def save_binary_confusion_matrix(y_true, y_pred, checkpoint_path, checkpoint_name, suffix="plot"):
    cm = confusion_matrix(y_true, y_pred)
    print("\n")
    df_cm = pd.DataFrame(cm, range(2), range(2))
    plt.clf()
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 16})
    plt.ylabel("Actual label", size=20)
    plt.xlabel("Predicted label", size=20)
    plt.xticks(np.arange(2), ["Fake", "Real"], size=16)
    plt.yticks(np.arange(2), ["Fake", "Real"], size=16)
    plt.ylim([2, 0])
    plt.tight_layout()
    plt.savefig(_checkpoint_output_path(checkpoint_path, checkpoint_name, suffix))
    calculated_acc = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
    print("Calculated Accuracy", calculated_acc * 100)

    y_true = ["Fake"] * sum(cm[0]) + ["Real"] * sum(cm[1])
    y_pred = (
        ["Fake"] * cm[0][0]
        + ["Real"] * cm[0][1]
        + ["Fake"] * cm[1][0]
        + ["Real"] * cm[1][1]
    )

    print("📊 Confusion Matrix:\n", cm)
    print("\n📈 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Fake", "Real"]))


def save_method_confusion_matrix(y_true_method, y_pred_method, checkpoint_path, checkpoint_name, suffix="plot(method)"):
    labels = ["original", "Deepfakes", "FaceShifter", "FaceSwap", "NeuralTextures", "Face2Face", "others"]
    label_indices = list(range(len(labels)))
    cm = confusion_matrix(y_true_method, y_pred_method, labels=label_indices)
    print("\n")
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.clf()
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 12}, cmap="Blues")
    plt.ylabel("Actual label", size=16)
    plt.xlabel("Predicted label", size=16)
    plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45, ha="right", fontsize=12)
    plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(_checkpoint_output_path(checkpoint_path, checkpoint_name, suffix))

    calculated_acc = np.trace(cm) / np.sum(cm)
    print(f"\n✅ Calculated Accuracy: {calculated_acc * 100:.2f}%")
    print("📊 Confusion Matrix:\n", cm)
    print("\n📈 Classification Report:")
    print(classification_report(y_true_method, y_pred_method, target_names=labels, labels=label_indices))


def save_roc_curve(
    true_bin,
    output_bin,
    checkpoint_path,
    checkpoint_name,
    suffix="roc_curve",
    score_mode="logits",
    print_auc=True,
    print_suffix=None,
):
    if score_mode == "logits":
        pred_score = torch.softmax(output_bin, dim=1)[:, 1].cpu().numpy()
    elif score_mode == "probability":
        pred_score = output_bin.cpu().numpy()
    else:
        raise ValueError(f"Unsupported score_mode: {score_mode}")

    fpr, tpr, _ = roc_curve(true_bin, pred_score)
    roc_auc = auc(fpr, tpr)
    if print_auc:
        print(f"✅ ROC curve (AUC = {roc_auc:.2f})")

    plt.clf()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Binary Classification)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    output_path = _checkpoint_output_path(checkpoint_path, checkpoint_name, suffix)
    plt.savefig(output_path)
    printed_suffix = print_suffix if print_suffix is not None else suffix
    print(f"✅ ROC Curve saved to {checkpoint_path}/{checkpoint_name}_{printed_suffix}.png")


def save_tsne_plot(
    features,
    labels,
    checkpoint_path,
    checkpoint_name,
    suffix="tsne",
    method_labels=None,
    title="t-SNE of Method Class Features",
    random_state=42,
):
    n_samples = features.shape[0]
    perplexity = min(30, max(5, n_samples // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    features = np.array(features)
    labels = np.array(labels)
    x_embedded = tsne.fit_transform(features)

    plt.clf()
    plt.figure(figsize=(8, 6))
    for cls in np.unique(labels):
        idx = labels == cls
        label_name = method_labels[cls] if method_labels and cls < len(method_labels) else str(cls)
        plt.scatter(x_embedded[idx, 0], x_embedded[idx, 1], label=label_name, alpha=0.7)

    plt.legend()
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(_checkpoint_output_path(checkpoint_path, checkpoint_name, suffix))
    print(f"✅ t-SNE plot saved to {checkpoint_path}/{checkpoint_name}_{suffix}.png")


def save_binary_tsne_plot(features, labels, checkpoint_path, checkpoint_name, suffix="tsne_binary(test)", random_state=42):
    n_samples = features.shape[0]
    perplexity = min(30, max(5, n_samples // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    features = np.array(features)
    labels = np.array(labels)
    x_embedded = tsne.fit_transform(features)

    plt.clf()
    plt.figure(figsize=(8, 6))
    for cls in np.unique(labels):
        idx = labels == cls
        label_name = "REAL" if cls == 1 else "FAKE"
        plt.scatter(x_embedded[idx, 0], x_embedded[idx, 1], label=label_name, alpha=0.7)

    plt.legend()
    plt.title("t-SNE of Binary Classification Features")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(_checkpoint_output_path(checkpoint_path, checkpoint_name, suffix))
    print(f"✅ t-SNE (Binary) plot saved to {checkpoint_path}/{checkpoint_name}_{suffix}.png")
