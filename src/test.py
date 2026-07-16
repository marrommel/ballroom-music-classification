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

# IMPORTANT: Adjust "inference" if your original script is named differently (e.g., "main.py")
from inference import (
    load_model,
    predict,
    DANCE_CLASSES,
    CHECKPOINT_PATH, extract_chunks
)


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


def plot_confusion_matrix(y_true, y_pred, output_path: str):
    """Generates and saves an interactive confusion matrix heatmap as HTML."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(DANCE_CLASSES)))

    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale='Blues',
        x=DANCE_CLASSES,
        y=DANCE_CLASSES,
        labels=dict(x="Predicted Class", y="True Class", color="Count"),
        title='Dance Style Confusion Matrix'
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
    # Setup directories
    audio_dir = Path("test_data_set")
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_dir.exists():
        print(f"Error: Directory '{audio_dir}' not found.")
        return

    # Gather audio files
    wav_files = list(audio_dir.rglob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in '{audio_dir}'.")
        return

    print(f"Loading model from: {CHECKPOINT_PATH}")
    model = load_model(CHECKPOINT_PATH)

    print(f"Evaluating {len(wav_files)} files...")
    y_true_indices, y_pred_indices, y_scores_list = [], [], []
    for filepath in tqdm(wav_files):
        true_class = get_true_label(filepath)

        if true_class is None:
            tqdm.write(f"Warning: Could not infer ground truth for {filepath.name}. Skipping.")
            continue

        # Extract features
        try:
            chunks = extract_chunks(str(filepath))
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

    if not y_true_indices:
        print("No valid files were successfully processed. Exiting.")
        return

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

    # Saving as .html files for interactivity
    cm_path = output_dir / "confusion_matrix.html"
    plot_confusion_matrix(y_true_indices, y_pred_indices, str(cm_path))

    roc_path = output_dir / "roc_curves.html"
    plot_roc_curves(y_true_indices, y_scores_list, str(roc_path))

    print("Done! Open the .html files in your browser to view the interactive plots.")


if __name__ == "__main__":
    main()