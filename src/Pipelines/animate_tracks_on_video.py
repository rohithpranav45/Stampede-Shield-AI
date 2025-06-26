import cv2
import pandas as pd
import numpy as np
import imageio
import os
from scipy.spatial.distance import cdist

def get_color(id):
    np.random.seed(int(id))
    return tuple(map(int, (np.random.rand(3,) * 255)))

def animate_tracks_on_video(video_path, traj_file, output_file="tracks_on_video.gif", delay=0.2, min_track_length=10, overlap_thresh=10):
    # Load trajectory data
    if traj_file.endswith('.parquet'):
        df = pd.read_parquet(traj_file)
    else:
        df = pd.read_csv(traj_file)
    # Filter out short tracks
    track_lengths = df.groupby('id').size()
    valid_ids = track_lengths[track_lengths >= min_track_length].index
    df = df[df['id'].isin(valid_ids)]
    ids = df['id'].unique()
    id_colors = {id_: get_color(id_) for id_ in ids}
    frame_nums = sorted(df['frame'].unique())

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Get all tracks up to this frame
        current_tracks = df[df['frame'] <= frame_idx]
        # For current frame, get last position of each id
        frame_tracks = df[df['frame'] == frame_idx]
        # Suppress overlapping points
        if not frame_tracks.empty:
            positions = frame_tracks[['x', 'y']].values
            ids_in_frame = frame_tracks['id'].values
            # Keep track of which ids to draw
            keep_ids = set()
            if len(positions) > 0:
                # Sort ids by track length (descending)
                id_lengths = {id_: track_lengths[id_] for id_ in ids_in_frame}
                sorted_ids = sorted(ids_in_frame, key=lambda x: -id_lengths[x])
                taken = np.zeros(len(positions), dtype=bool)
                for i, id_ in enumerate(sorted_ids):
                    if taken[i]:
                        continue
                    keep_ids.add(id_)
                    # Mark all close points as taken
                    dists = np.linalg.norm(positions - positions[i], axis=1)
                    close = (dists < overlap_thresh)
                    taken = taken | close
        else:
            keep_ids = set()
        for id_ in ids:
            if id_ not in keep_ids:
                continue
            track = current_tracks[current_tracks['id'] == id_]
            if len(track) == 0:
                continue
            color = id_colors[id_]
            pts = track[['x', 'y']].values.astype(int)
            # Draw path
            for i in range(1, len(pts)):
                cv2.line(frame, tuple(pts[i-1]), tuple(pts[i]), color, 2)
            # Draw current position
            cv2.circle(frame, tuple(pts[-1]), 6, color, -1)
            cv2.circle(frame, tuple(pts[-1]), 8, (255,255,255), 2)
        # Convert BGR to RGB for imageio
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_idx += 1
        print(f"Frame {frame_idx} done")
    cap.release()

    # Save as GIF
    imageio.mimsave(output_file, frames, fps=1/delay, loop=0)
    print(f"Tracking animation saved as {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Animate tracks overlayed on original video frames.")
    parser.add_argument('--video', type=str, required=True, help='Input video file')
    parser.add_argument('--traj_file', type=str, required=True, help='Trajectory Parquet or CSV file (from DeepSORT logging)')
    parser.add_argument('--output', type=str, default='tracks_on_video.gif', help='Output GIF filename')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay (seconds) between frames')
    parser.add_argument('--min_track_length', type=int, default=10, help='Minimum number of frames for a track to be visualized')
    parser.add_argument('--overlap_thresh', type=float, default=10, help='Distance threshold for suppressing overlapping points (pixels)')
    args = parser.parse_args()
    animate_tracks_on_video(args.video, args.traj_file, output_file=args.output, delay=args.delay, min_track_length=args.min_track_length, overlap_thresh=args.overlap_thresh) 