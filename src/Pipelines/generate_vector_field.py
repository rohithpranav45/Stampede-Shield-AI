import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def compute_and_plot_velocity_field(traj_file, frame=None, grid_size=20, output_img='vector_field.png'):
    # Load trajectory data
    if traj_file.endswith('.parquet'):
        df = pd.read_parquet(traj_file)
    else:
        df = pd.read_csv(traj_file)
    # Sort and compute velocities
    frame_col = 'frame' if 'frame' in df.columns else 'step'
    df = df.sort_values(['id', frame_col])
    df['vx'] = df.groupby('id')[['x']].diff()
    df['vy'] = df.groupby('id')[['y']].diff()
    df['dt'] = df.groupby('id')[frame_col].diff()
    df['vx'] = df['vx'] / df['dt']
    df['vy'] = df['vy'] / df['dt']
    # Use last frame if not specified
    if frame is None:
        frame = df[frame_col].max()
    points = df[df[frame_col] == frame][['x', 'y', 'vx', 'vy']].dropna().values
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    x_edges = np.linspace(x_min, x_max, grid_size+1)
    y_edges = np.linspace(y_min, y_max, grid_size+1)
    # Compute average velocity per cell
    U = np.zeros((grid_size, grid_size))
    V = np.zeros((grid_size, grid_size))
    C = np.zeros((grid_size, grid_size))
    for x, y, vx, vy in points:
        ix = np.searchsorted(x_edges, x, side='right') - 1
        iy = np.searchsorted(y_edges, y, side='right') - 1
        if 0 <= ix < grid_size and 0 <= iy < grid_size:
            U[ix, iy] += vx
            V[ix, iy] += vy
            C[ix, iy] += 1
    # Avoid division by zero
    mask = C > 0
    U[mask] /= C[mask]
    V[mask] /= C[mask]
    # Grid centers
    Xc = (x_edges[:-1] + x_edges[1:]) / 2
    Yc = (y_edges[:-1] + y_edges[1:]) / 2
    X, Y = np.meshgrid(Xc, Yc, indexing='ij')
    plt.figure(figsize=(8,8))
    plt.quiver(X, Y, U, V, C, cmap='viridis', scale=1, scale_units='xy', width=0.005)
    plt.colorbar(label='Number of people in cell')
    plt.title(f'Vector Field (Frame {frame})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()
    print(f"Vector field saved as {output_img}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and plot average velocity vectors per grid cell.')
    parser.add_argument('--traj_file', type=str, required=True, help='Trajectory file (parquet or csv)')
    parser.add_argument('--frame', type=int, default=None, help='Frame number (default: last frame)')
    parser.add_argument('--grid_size', type=int, default=20, help='Number of grid cells per axis')
    parser.add_argument('--output_img', type=str, default='vector_field.png', help='Output image filename')
    args = parser.parse_args()
    compute_and_plot_velocity_field(args.traj_file, frame=args.frame, grid_size=args.grid_size, output_img=args.output_img) 