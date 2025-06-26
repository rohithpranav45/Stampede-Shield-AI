import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import get_grid_edges, plot_density_field

def generate_density_field(traj_file, frame=None, grid_size=40, output_img='density_field.png'):
    # Load trajectory data
    if traj_file.endswith('.parquet'):
        df = pd.read_parquet(traj_file)
    else:
        df = pd.read_csv(traj_file)
    # Use last frame if not specified
    frame_col = 'frame' if 'frame' in df.columns else 'step'
    if frame is None:
        frame = df[frame_col].max()
    points = df[df[frame_col] == frame][['x', 'y']].values
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    x_edges, y_edges = get_grid_edges(x_min, x_max, y_min, y_max, grid_size)
    H, _, _ = np.histogram2d(points[:,0], points[:,1], bins=[x_edges, y_edges])
    plot_density_field(H, x_edges, y_edges, output_img, title=f'Density Field (Frame {frame})')
    print(f"Density field saved as {output_img}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate density field and grid overlay from trajectory file.')
    parser.add_argument('--traj_file', type=str, required=True, help='Trajectory file (parquet or csv)')
    parser.add_argument('--frame', type=int, default=None, help='Frame number (default: last frame)')
    parser.add_argument('--grid_size', type=int, default=20, help='Number of grid cells per axis')
    parser.add_argument('--output_img', type=str, default='density_field.png', help='Output image filename')
    args = parser.parse_args()
    generate_density_field(args.traj_file, frame=args.frame, grid_size=args.grid_size, output_img=args.output_img) 