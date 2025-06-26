import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio

def get_color(id):
    np.random.seed(int(id))
    return np.random.rand(3,)

def animate_topview_tracks(traj_file, output_gif="topview_tracks.gif", delay=0.2):
    # Load trajectory data
    if traj_file.endswith('.parquet'):
        df = pd.read_parquet(traj_file)
    else:
        df = pd.read_csv(traj_file)
    ids = df['id'].unique()
    frames = []
    all_x, all_y = df['x'].values, df['y'].values
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    frame_nums = sorted(df['frame'].unique())
    id_colors = {id_: get_color(id_) for id_ in ids}

    for f in frame_nums:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('black')
        for id_ in ids:
            track = df[(df['id'] == id_) & (df['frame'] <= f)]
            if len(track) == 0:
                continue
            color = id_colors[id_]
            # Draw path
            ax.plot(track['x'], track['y'], color=color, linewidth=2)
            # Draw current position
            ax.scatter(track['x'].iloc[-1], track['y'].iloc[-1], color=color, s=40, edgecolor='white', zorder=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        plt.tight_layout(pad=0)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(image)
        plt.close(fig)
        print(f"Frame {f} done")
    imageio.mimsave(output_gif, frames, fps=1/delay, loop=0)
    print(f"Top-view tracking animation saved as {output_gif}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Animate 2D top-view with tracking lines for each ID (DeepSORT).")
    parser.add_argument('--traj_file', type=str, required=True, help='Trajectory Parquet or CSV file (from DeepSORT logging)')
    parser.add_argument('--output_gif', type=str, default='topview_tracks.gif', help='Output GIF filename')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay (seconds) between frames')
    args = parser.parse_args()
    animate_topview_tracks(args.traj_file, output_gif=args.output_gif, delay=args.delay) 