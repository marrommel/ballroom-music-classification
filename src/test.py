from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    top_k_accuracy_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from config import Config
from inference import load_model, predict, extract_chunks

_config = Config()
DANCE_CLASSES = _config.dance_classes

def get_true_label(filepath: Path) -> str:
    """
    Extract the ground truth label from the file path.
    Assumes either folder name or file name contains the class name
    (e.g., 'audio/Waltz/01.wav' or 'audio/Tango_song.wav').
    """
    path_str = str(filepath).lower()
    for cls in DANCE_CLASSES:
        if cls.lower() in path_str:
            return cls
    return None


def plot_confusion_matrix(y_true, y_pred, output_path: str, normalize: bool = False):
    """Generates and saves an interactive confusion matrix heatmap as HTML.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the HTML file
        normalize: If True, display percentages; if False, display absolute counts
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(DANCE_CLASSES)))

    if normalize:
        # Convert to percentages
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        text_format = '.2f'
        title = 'Dance Style Confusion Matrix (Percentage)'
    else:
        cm_display = cm
        text_format = 'd'
        title = 'Dance Style Confusion Matrix (Absolute)'

    fig = px.imshow(
        cm_display,
        text_auto=text_format,
        color_continuous_scale='Blues',
        x=DANCE_CLASSES,
        y=DANCE_CLASSES,
        labels=dict(x="Predicted Class", y="True Class", color="Count" if not normalize else "Percentage (%)"),
        title=title
    )

    fig.update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        xaxis_tickangle=-45
    )

    fig.write_html(output_path)


def plot_roc_curves(y_true, y_scores, output_path: str):
    """Generates and saves interactive One-vs-Rest ROC curves as HTML."""
    # Binarize labels for One-vs-Rest ROC calculation
    y_bin = label_binarize(y_true, classes=range(len(DANCE_CLASSES)))
    y_scores = np.array(y_scores)

    fig = go.Figure()

    for i, cls in enumerate(DANCE_CLASSES):
        # Only plot if the class is actually present in the ground truth
        if np.sum(y_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{cls} (AUC = {roc_auc:.2f})',
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
            ))

    # Add diagonal dashed line for chance performance
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='black'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) - One-vs-Rest',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0.0, 1.0], constrain='domain'),
        yaxis=dict(range=[0.0, 1.05], scaleanchor="x", scaleratio=1),
        legend=dict(x=0.7, y=0.1, bordercolor="Black", borderwidth=1)
    )

    fig.write_html(output_path)


def main():
    config = Config()

    # Setup directories
    test_data_dir = Path("assets/test_data_set")
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_data_dir.exists():
        print(f"Error: Directory '{test_data_dir}' not found.")
        return

    # Gather audio files
    wav_files = list(test_data_dir.rglob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in '{test_data_dir}'.")
        return

    print(f"Loading model from: {config.inference_model_weights}")
    model = load_model(config.inference_model_weights)

    print(f"Evaluating {len(wav_files)} files...")
    processed_files, y_true_indices, y_pred_indices, y_scores_list = [], [], [], []
    for filepath in tqdm(wav_files):
        true_class = get_true_label(filepath)

        if true_class is None:
            tqdm.write(f"Warning: Could not infer ground truth for {filepath.name}. Skipping.")
            continue

        # Extract features
        try:
            chunks = extract_chunks(str(filepath), config.spec_types)
        except ValueError as e:
            tqdm.write(f"Skipping {filepath.name}: {e}")
            continue

        if not chunks:
            tqdm.write(f"Skipping {filepath.name}: no chunks extracted (audio too short).")
            continue

        # Inference
        predicted_class, prob_map = predict(model, chunks)
        probs = [prob_map[cls] for cls in DANCE_CLASSES]

        # Store for metrics
        y_true_indices.append(DANCE_CLASSES.index(true_class))
        y_pred_indices.append(DANCE_CLASSES.index(predicted_class))
        y_scores_list.append(probs)
        processed_files.append(filepath)  # <-- add this

    if not y_true_indices:
        print("No valid files were successfully processed. Exiting.")
        return

    wrong_by_class = {cls: [] for cls in DANCE_CLASSES}
    for filepath, true_idx, pred_idx in zip(processed_files, y_true_indices, y_pred_indices):
        if true_idx != pred_idx:
            true_cls = DANCE_CLASSES[true_idx]
            pred_cls = DANCE_CLASSES[pred_idx]
            wrong_by_class[true_cls].append((filepath.stem, pred_cls))

    print("\n" + "=" * 40)
    print("WRONG PREDICTIONS PER DANCE CLASS")
    print("=" * 40)
    for cls in DANCE_CLASSES:
        wrongs = wrong_by_class[cls]
        if wrongs:
            print(f"\n[{cls}] — {len(wrongs)} wrong prediction(s):")
            for song_name, predicted_as in wrongs:
                print(f"  '{song_name}'  →  predicted as: {predicted_as}")

    # ---------- Calculate Metrics ----------
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)

    top1_acc = accuracy_score(y_true_indices, y_pred_indices)
    print(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")

    # Top-2 accuracy requires true labels, probability scores, and class ordering
    try:
        top2_acc = top_k_accuracy_score(
            y_true_indices, y_scores_list, k=2, labels=range(len(DANCE_CLASSES))
        )
        print(f"Top-2 Accuracy: {top2_acc * 100:.2f}%")
    except ValueError:
        print("Top-2 Accuracy: Not enough classes predicted/present to compute.")

    print("\nClassification Report:")
    report = classification_report(
        y_true_indices, y_pred_indices,
        target_names=DANCE_CLASSES,
        labels=range(len(DANCE_CLASSES)),
        zero_division=0
    )
    print(report)

    # ---------- Generate Plots ----------
    print(f"\nGenerating interactive plots in '{output_dir.absolute()}'...")

    # Absolute confusion matrix
    cm_abs_path = output_dir / "confusion_matrix_absolute.html"
    plot_confusion_matrix(y_true_indices, y_pred_indices, str(cm_abs_path), normalize=False)

    # Relative (percentage) confusion matrix
    cm_rel_path = output_dir / "confusion_matrix_relative.html"
    plot_confusion_matrix(y_true_indices, y_pred_indices, str(cm_rel_path), normalize=True)

    roc_path = output_dir / "roc_curves.html"
    plot_roc_curves(y_true_indices, y_scores_list, str(roc_path))

    print("Done! Open the .html files in your browser to view the interactive plots.")


if __name__ == "__main__":
    main()