import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Simple SPH kernel for demonstration (Gaussian)
def sph_kernel(r, h):
    return np.exp(-0.5 * (r / h) ** 2)

def forecast_sph_trajectories(
    traj_file, 
    time_window=10, 
    frame_interval=0.2, 
    history_window=5, 
    forecast_steps=10, 
    dt=1, 
    h=5, 
    output_img='sph_forecast.png', 
    output_csv='sph_forecast_positions.csv',
    frame=None
):
    # Load trajectory data
    if traj_file.endswith('.parquet'):
        df = pd.read_parquet(traj_file)
    else:
        df = pd.read_csv(traj_file)
    df = df.sort_values(['id', 'frame'])
    # Use last frame if not specified
    if frame is None:
        frame = df['frame'].max()
    # For each person, use last N (history_window) frames to estimate velocity
    last_frames = df.groupby('id').tail(history_window)
    velocities = last_frames.groupby('id').apply(
        lambda g: pd.Series({
            'x': g['x'].iloc[-1],
            'y': g['y'].iloc[-1],
            'vx': np.mean(g['x'].diff().iloc[1:] / g['frame'].diff().iloc[1:]),
            'vy': np.mean(g['y'].diff().iloc[1:] / g['frame'].diff().iloc[1:])
        })
    ).reset_index()
    ids = velocities['id'].values
    positions = velocities[['x', 'y']].values
    velocities_arr = velocities[['vx', 'vy']].values
    # SPH-like forecast: for each particle, update position using weighted average of neighbor velocities
    pred_positions = positions.copy()
    pred_velocities = velocities_arr.copy()
    # If forecast_steps not set, compute from time_window and dt
    if forecast_steps is None:
        forecast_steps = int(time_window / dt)
    for step in range(forecast_steps):
        new_positions = pred_positions.copy()
        new_velocities = pred_velocities.copy()
        for i, pos in enumerate(pred_positions):
            r = np.linalg.norm(pred_positions - pos, axis=1)
            w = sph_kernel(r, h)
            v_avg = np.average(pred_velocities, axis=0, weights=w)
            new_velocities[i] = v_avg
            new_positions[i] = pos + v_avg * dt
        pred_positions = new_positions
        pred_velocities = new_velocities
    # Save as CSV
    out_df = pd.DataFrame({'id': ids, 'future_x': pred_positions[:,0], 'future_y': pred_positions[:,1]})
    out_df.to_csv(output_csv, index=False)
    print(f"SPH-forecasted future positions saved as {output_csv}")
    # Plot
    plt.figure(figsize=(8,8))
    plt.scatter(positions[:,0], positions[:,1], color='blue', label='Current', alpha=0.6)
    plt.scatter(pred_positions[:,0], pred_positions[:,1], color='red', label=f'Forecast ({time_window}s)', alpha=0.6)
    for i in range(len(positions)):
        plt.arrow(positions[i,0], positions[i,1], pred_positions[i,0]-positions[i,0], pred_positions[i,1]-positions[i,1],
                  color='green', head_width=0.5, head_length=1, alpha=0.5, length_includes_head=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'SPH/Fluid Forecasted Top-View after {time_window}s')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_img)
    plt.close()
    print(f"SPH/Fluid forecast topview saved as {output_img}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SPH/Fluid-based forecasting of pedestrian flow using full trajectory history.')
    parser.add_argument('--traj_file', type=str, required=True, help='Trajectory file (parquet or csv)')
    parser.add_argument('--time_window', type=float, default=10, help='Time window in seconds')
    parser.add_argument('--frame_interval', type=float, default=0.2, help='Frame interval in seconds')
    parser.add_argument('--history_window', type=int, default=5, help='Number of frames to use for velocity estimation')
    parser.add_argument('--forecast_steps', type=int, default=None, help='Number of forecast steps (default: time_window/dt)')
    parser.add_argument('--dt', type=float, default=1, help='Time step for forecast')
    parser.add_argument('--h', type=float, default=5, help='SPH kernel bandwidth')
    parser.add_argument('--output_img', type=str, default='sph_forecast.png', help='Output image filename')
    parser.add_argument('--output_csv', type=str, default='sph_forecast_positions.csv', help='Output CSV filename')
    parser.add_argument('--frame', type=int, default=None, help='Frame number (default: last frame)')
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
        output_csv=args.output_csv,
        frame=args.frame
    ) 