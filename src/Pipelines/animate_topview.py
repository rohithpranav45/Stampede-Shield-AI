import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

def animate_topview(checkpoints_dir, output_gif="topview.gif", delay=0.2, dot_size=10):
    csv_files = sorted(glob.glob(os.path.join(checkpoints_dir, "*.csv")))
    if not csv_files:
        print("No CSV files found in", checkpoints_dir)
        return

    frames = []
    all_x, all_y = [], []

    # First, find global x/y limits for consistent axes
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_x.extend(df['x'].values)
        all_y.extend(df['y'].values)
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        x = df['x'].values
        y = df['y'].values

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('black')
        ax.scatter(x, y, c='white', s=dot_size)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        plt.tight_layout(pad=0)
        # Save frame to buffer
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())
        image = image[..., :3]  # Drop alpha channel for GIF
        frames.append(image)
        plt.close(fig)
        print(f"Processed {os.path.basename(csv_file)}")

    # Save as GIF
    imageio.mimsave(output_gif, frames, fps=1/delay, loop=0)
    print(f"Top-view animation saved as {output_gif}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Animate 2D top-view of crowd checkpoints (x, y) as a GIF.")
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Directory with checkpoint CSV files')
    parser.add_argument('--output_gif', type=str, default='topview.gif', help='Output GIF filename')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay (seconds) between frames')
    parser.add_argument('--dot_size', type=int, default=10, help='Dot size for each point')
    args = parser.parse_args()

    animate_topview(args.checkpoints_dir, output_gif=args.output_gif, delay=args.delay, dot_size=args.dot_size)
