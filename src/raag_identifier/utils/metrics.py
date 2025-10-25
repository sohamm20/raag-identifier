"""
Evaluation metrics and reporting for Raag classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from pathlib import Path
from typing import List, Dict, Optional
import json


class MetricsCalculator:
    """
    Calculate and report classification metrics.
    """

    def __init__(self, class_names: List[str]):
        """
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.n_classes = len(class_names)

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """
        Calculate all metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of metrics
        """
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(self.n_classes), average=None
        )

        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=range(self.n_classes), average='macro'
        )

        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=range(self.n_classes), average='weighted'
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.n_classes))

        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'per_class': {},
            'confusion_matrix': cm.tolist(),
        }

        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i]),
            }

        return metrics

    def print_metrics(self, metrics: Dict):
        """
        Print metrics in a readable format.

        Args:
            metrics: Metrics dictionary
        """
        print("=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        print(f"\nMacro Averages:")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")
        print(f"  F1-Score:  {metrics['f1_macro']:.4f}")
        print(f"\nWeighted Averages:")
        print(f"  Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Recall:    {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score:  {metrics['f1_weighted']:.4f}")

        print("\n" + "-" * 60)
        print("Per-Class Metrics:")
        print("-" * 60)
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 60)

        for class_name, class_metrics in metrics['per_class'].items():
            print(
                f"{class_name:<20} "
                f"{class_metrics['precision']:<12.4f} "
                f"{class_metrics['recall']:<12.4f} "
                f"{class_metrics['f1']:<12.4f} "
                f"{class_metrics['support']:<10}"
            )

        print("=" * 60)

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        save_path: Optional[str] = None,
        normalize: bool = False,
    ):
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            save_path: Path to save plot
            normalize: Whether to normalize by true labels
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = '.2f'
        else:
            fmt = 'd'

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
        )

        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.close()

    def save_metrics_report(
        self,
        metrics: Dict,
        output_dir: str,
        prefix: str = 'test',
    ):
        """
        Save metrics report to files.

        Args:
            metrics: Metrics dictionary
            output_dir: Output directory
            prefix: Prefix for output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        json_path = output_dir / f'{prefix}_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {json_path}")

        # Save CSV report (per-class metrics)
        csv_path = output_dir / f'{prefix}_metrics.csv'
        df_data = []
        for class_name, class_metrics in metrics['per_class'].items():
            row = {'class': class_name}
            row.update(class_metrics)
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
        print(f"Per-class metrics saved to {csv_path}")

        # Save confusion matrix plot
        cm = np.array(metrics['confusion_matrix'])
        cm_path = output_dir / f'{prefix}_confusion_matrix.png'
        self.plot_confusion_matrix(cm, save_path=str(cm_path))

        # Save normalized confusion matrix
        cm_norm_path = output_dir / f'{prefix}_confusion_matrix_normalized.png'
        self.plot_confusion_matrix(cm, save_path=str(cm_norm_path), normalize=True)


def plot_training_history(
    history: Dict,
    save_path: Optional[str] = None,
):
    """
    Plot training history (loss and accuracy curves).

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")

    plt.close()
