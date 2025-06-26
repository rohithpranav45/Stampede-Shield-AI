import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def get_key_steps(traj_csv, monitor_dir):
    # Default values
    panic_trigger_step = 0
    first_stampede_step = None
    # Try to read panic_trigger_step from config or monitor output
    try:
        import yaml
        with open('pipeline_config.yaml') as f:
            cfg = yaml.safe_load(f)
        panic_trigger_step = cfg.get('panic_simulation', {}).get('panic_trigger_step', 0)
    except Exception:
        pass
    # Try to read first_stampede step from monitor output
    try:
        df_alerts = pd.read_csv(os.path.join(monitor_dir, 'pressure_alerts.csv'))
        if not df_alerts.empty:
            first_stampede_step = df_alerts['start_step'].min()
    except Exception:
        pass
    return panic_trigger_step, first_stampede_step

def plot_frame(ax, frame_df, x_min, x_max, y_min, y_max, event_text=None):
    colors = frame_df['panic'].map({True: 'red', False: 'gray'})
    ax.scatter(frame_df['x'], frame_df['y'], c=colors, s=40, edgecolor='white')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if event_text:
        ax.set_title(event_text, fontsize=14, color='black')
    else:
        ax.set_title('')

def animate_key_frames(traj_csv, monitor_dir, output_gif='key_frames_animation.gif'):
    df = pd.read_csv(traj_csv)
    steps = df['step'].unique()
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    # Get key steps
    panic_trigger_step, first_stampede_step = get_key_steps(traj_csv, monitor_dir)
    key_steps = [0]
    if panic_trigger_step not in key_steps:
        key_steps.append(panic_trigger_step)
    if first_stampede_step is not None and first_stampede_step not in key_steps:
        key_steps.append(first_stampede_step)
    if steps.max() not in key_steps:
        key_steps.append(steps.max())
    key_steps = sorted(set(key_steps))
    # Prepare animation
    import imageio
    frames = []
    for step in key_steps:
        frame_df = df[df['step'] == step]
        fig, ax = plt.subplots(figsize=(8,8))
        event_text = None
        if step == 0:
            event_text = f'Start (t=0s)'
        elif step == panic_trigger_step:
            event_text = f'Panic Injection (t={step})'
        elif first_stampede_step is not None and step == first_stampede_step:
            event_text = f'First Stampede Detected (t={step})'
        elif step == steps.max():
            event_text = f'End (t={step})'
        plot_frame(ax, frame_df, x_min, x_max, y_min, y_max, event_text)
        plt.tight_layout()
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(image)
        plt.close(fig)
    # Save GIF
    imageio.mimsave(output_gif, frames, fps=1, loop=0)
    print(f'Key frames animation saved as {output_gif}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Animate key frames (start, panic injection, first stampede, end) in top-view point cloud.')
    parser.add_argument('--traj_csv', type=str, default='panic_sfm_trajectories.csv', help='Trajectory CSV file')
    parser.add_argument('--monitor_dir', type=str, default='panic_monitor_frames', help='Directory with monitor outputs')
    parser.add_argument('--output_gif', type=str, default='key_frames_animation.gif', help='Output GIF filename')
    args = parser.parse_args()
    animate_key_frames(args.traj_csv, args.monitor_dir, args.output_gif) 