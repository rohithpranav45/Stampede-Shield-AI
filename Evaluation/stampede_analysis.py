import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_simulated_roc(output=None):
    """Plot a simulated ROC curve for anomaly (stampede) detection."""
    fpr = [0.0, 0.1, 0.2, 0.4, 0.6, 1.0]
    tpr = [0.0, 0.6, 0.75, 0.88, 0.95, 1.0]
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – Anomaly Detection')
    plt.legend(loc="lower right")
    plt.grid(True)
    if output:
        plt.savefig(output, bbox_inches='tight')
        print(f"ROC curve saved to {output}")
    else:
        plt.show()
    plt.close()


def plot_roc_from_labels(labels_file, scores_file, output=None):
    """Plot ROC curve from true labels and predicted anomaly scores."""
    y_true = np.loadtxt(labels_file, delimiter=',')
    y_score = np.loadtxt(scores_file, delimiter=',')
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve – Anomaly Detection')
    plt.legend(loc="lower right")
    plt.grid(True)
    if output:
        plt.savefig(output, bbox_inches='tight')
        print(f"ROC curve saved to {output}")
    else:
        plt.show()
    plt.close()


def plot_heatmap(data_file, title='Crowd Density Heatmap', cmap='hot', output=None):
    """Visualize a density or flow heatmap from a CSV or npy file."""
    if data_file.endswith('.csv'):
        data = pd.read_csv(data_file, header=None).values
    elif data_file.endswith('.npy'):
        data = np.load(data_file)
    else:
        raise ValueError('Unsupported file format. Use .csv or .npy')
    plt.figure(figsize=(8, 7))
    im = plt.imshow(data, cmap=cmap, origin='upper')
    plt.colorbar(im, label='People Count' if 'Density' in title else 'Flow Magnitude')
    plt.title(title)
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    # Annotate cells
    for (i, j), val in np.ndenumerate(data):
        plt.text(j, i, int(val), ha='center', va='center', color='white' if val > data.max()/2 else 'black', fontsize=10)
    plt.tight_layout()
    if output:
        plt.savefig(output, bbox_inches='tight')
        print(f"Heatmap saved to {output}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Stampede Analysis: ROC curve and density/flow heatmap visualization.')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # ROC curve mode
    roc_parser = subparsers.add_parser('roc', help='Plot ROC curve for anomaly/stampede detection')
    roc_parser.add_argument('--simulate', action='store_true', help='Plot a simulated ROC curve')
    roc_parser.add_argument('--labels', type=str, help='CSV file with true labels (0/1)')
    roc_parser.add_argument('--scores', type=str, help='CSV file with predicted anomaly scores')
    roc_parser.add_argument('--output', type=str, help='Output image file (optional)')

    # Heatmap mode
    heatmap_parser = subparsers.add_parser('heatmap', help='Visualize density or flow heatmap')
    heatmap_parser.add_argument('--data', type=str, required=True, help='CSV or npy file with density or flow matrix')
    heatmap_parser.add_argument('--title', type=str, default='Crowd Density Heatmap', help='Plot title')
    heatmap_parser.add_argument('--cmap', type=str, default='hot', help='Colormap (default: hot)')
    heatmap_parser.add_argument('--output', type=str, help='Output image file (optional)')

    args = parser.parse_args()

    if args.mode == 'roc':
        if args.simulate:
            plot_simulated_roc(args.output)
        elif args.labels and args.scores:
            plot_roc_from_labels(args.labels, args.scores, args.output)
        else:
            print('Specify either --simulate or both --labels and --scores for ROC mode.')
    elif args.mode == 'heatmap':
        plot_heatmap(args.data, title=args.title, cmap=args.cmap, output=args.output)


if __name__ == '__main__':
    main() 