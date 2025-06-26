import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def forecast_sph_trajectories(
    traj_file, 
    time_window=10, 
    frame_interval=0.2, 
    history_window=5, 
    forecast_steps=10, 
    dt=1, 
    h=5, 
    output_img='sph_forecast.png', 
    output_csv='sph_forecast_positions.csv'
):
    # Load trajectory data
    if traj_file.endswith('.parquet'):
        df = pd.read_parquet(traj_file)
    else:
        df = pd.read_csv(traj_file)
    # Sort and compute velocities
    df = df.sort_values(['id', 'frame'])
    df['vx'] = df.groupby('id')['x'].diff() / df.groupby('id')['frame'].diff()
    df['vy'] = df.groupby('id')['y'].diff() / df.groupby('id')['frame'].diff()
    # Use last frame for each id
    last = df.groupby('id').tail(1).dropna(subset=['vx', 'vy'])
    # Predict future positions
    dt = time_window / frame_interval  # number of frames in the time window
    last['future_x'] = last['x'] + last['vx'] * dt
    last['future_y'] = last['y'] + last['vy'] * dt
    # Save as CSV
    last[['id', 'future_x', 'future_y']].to_csv(output_csv, index=False)
    print(f"Future positions saved as {output_csv}")
    # Plot
    plt.figure(figsize=(8,8))
    plt.scatter(last['x'], last['y'], color='blue', label='Current', alpha=0.6)
    plt.scatter(last['future_x'], last['future_y'], color='red', label=f'Future ({time_window}s)', alpha=0.6)
    for i in range(len(last)):
        plt.arrow(last['x'].iloc[i], last['y'].iloc[i], last['future_x'].iloc[i]-last['x'].iloc[i], last['future_y'].iloc[i]-last['y'].iloc[i],
                  color='green', head_width=0.5, head_length=1, alpha=0.5, length_includes_head=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Predicted Top-View Point Cloud after {time_window}s')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()
    print(f"Future topview point cloud saved as {output_img}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SPH-based forecasting of pedestrian flow.')
    parser.add_argument('--traj_file', type=str, required=True, help='Trajectory file (parquet or csv)')
    parser.add_argument('--frame', type=int, default=None, help='Frame number (default: last frame)')
    parser.add_argument('--time_window', type=float, default=10, help='Time window in seconds')
    parser.add_argument('--frame_interval', type=float, default=0.2, help='Frame interval in seconds')
    parser.add_argument('--history_window', type=int, default=5, help='Number of frames to use for velocity estimation')
    parser.add_argument('--forecast_steps', type=int, default=10, help='Number of forecast steps')
    parser.add_argument('--dt', type=float, default=1, help='Time step for forecast')
    parser.add_argument('--h', type=float, default=5, help='SPH kernel bandwidth')
    parser.add_argument('--output_img', type=str, default='sph_forecast.png', help='Output image filename')
    parser.add_argument('--output_csv', type=str, default='sph_forecast_positions.csv', help='Output CSV filename')
    args = parser.parse_args()
    forecast_sph_trajectories(
        args.traj_file, 
        time_window=args.time_window, 
        frame_interval=args.frame_interval, 
        history_window=args.history_window, 
        forecast_steps=args.forecast_steps, 
        dt=args.dt, 
        h=args.h, 
        output_img=args.output_img, 
        output_csv=args.output_csv
    )
