import pandas as pd
import numpy as np
import argparse

# Step 1: Initialize agents from the last frame of the trajectory file

def initialize_agents(traj_file, history_window=5, output_csv='panic_agents_init.csv'):
    # Load trajectory data
    if traj_file.endswith('.parquet'):
        df = pd.read_parquet(traj_file)
    else:
        df = pd.read_csv(traj_file)
    df = df.sort_values(['id', 'frame'])
    # Use last N frames for velocity estimation
    last_frames = df.groupby('id').tail(history_window)
    agents = last_frames.groupby('id').apply(
        lambda g: pd.Series({
            'id': g['id'].iloc[-1],
            'x': g['x'].iloc[-1],
            'y': g['y'].iloc[-1],
            'vx': np.mean(g['x'].diff().iloc[1:] / g['frame'].diff().iloc[1:]),
            'vy': np.mean(g['y'].diff().iloc[1:] / g['frame'].diff().iloc[1:]),
            'panic': False
        })
    ).reset_index(drop=True)
    agents.to_csv(output_csv, index=False)
    print(f"Initialized agent states saved to {output_csv}")
    return agents

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initialize agents for panic simulation from trajectory file.')
    parser.add_argument('--traj_file', type=str, required=True, help='Trajectory file (parquet or csv)')
    parser.add_argument('--history_window', type=int, default=5, help='Number of frames to use for velocity estimation')
    parser.add_argument('--output_csv', type=str, default='panic_agents_init.csv', help='Output CSV for initial agent states')
    args = parser.parse_args()
    initialize_agents(args.traj_file, history_window=args.history_window, output_csv=args.output_csv) 