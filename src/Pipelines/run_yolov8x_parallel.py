import os
import cv2
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import pandas as pd
df = pd.read_csv('checkpoints_pipeline/checkpoint_0061.csv')
print(df.columns)
def split_video(video_path, num_chunks, output_dir):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    chunk_size = total_frames // num_chunks
    os.makedirs(output_dir, exist_ok=True)
    chunk_paths = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else total_frames
        out_path = os.path.join(output_dir, f"chunk_{i}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        out = cv2.VideoWriter(out_path, fourcc, fps, (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ))
        for f in range(start, end):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        chunk_paths.append((out_path, start))
    cap.release()
    return chunk_paths

def process_chunk(args):
    chunk_path, start_frame, interval, checkpoints_dir, chunk_idx, gpu_id = args
    chunk_checkpoints = os.path.join(checkpoints_dir, f"chunk_{chunk_idx}")
    os.makedirs(chunk_checkpoints, exist_ok=True)
    traj_file = os.path.join(chunk_checkpoints, f"trajectories_{chunk_idx}.parquet")
    cmd = [
        "python", "run_yolov8x.py",
        "--video", chunk_path,
        "--interval", str(interval),
        "--checkpoints_dir", chunk_checkpoints
    ]
    print(f"Processing chunk {chunk_idx}...")
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    subprocess.run(cmd, check=True, env=env)
    # Add start_frame offset to frame numbers in the output
    if os.path.exists(traj_file):
        df = pd.read_parquet(traj_file)
        for col in ['frame', 'frame_id', 'frame_number', 'frameIndex', 'image_id', 'frame_idx']:
            if col in df.columns:
                frame_col = col
                break
        df['frame'] += start_frame
        df.rename(columns={'frame_id': 'frame'}, inplace=True)
        df.to_parquet(traj_file)
    return traj_file

def merge_trajectories(traj_files, output_file):
    dfs = [pd.read_parquet(f) for f in traj_files if os.path.exists(f)]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values(by=['frame', 'id'])
    df_all.to_parquet(output_file)
    print(f"Merged trajectories saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Parallel run_yolov8x on video chunks.")
    parser.add_argument('--video', type=str, required=True, help='Input video file')
    parser.add_argument('--interval', type=float, default=0.2, help='Interval for detection/tracking')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints_parallel', help='Directory for checkpoints')
    parser.add_argument('--num_chunks', type=int, default=4, help='Number of parallel chunks')
    parser.add_argument('--output_traj', type=str, default='trajectories_parallel.parquet', help='Merged output trajectory file')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated list of GPU IDs')
    args = parser.parse_args()

    # Step 1: Split video
    chunk_dir = os.path.join(args.checkpoints_dir, "chunks")
    chunk_paths = split_video(args.video, args.num_chunks, chunk_dir)

    # Step 2: Process chunks in parallel
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    process_args = [
        (chunk_path, start_frame, args.interval, args.checkpoints_dir, idx, gpu_id)
        for idx, (chunk_path, start_frame) in enumerate(chunk_paths)
        for gpu_id in gpu_ids
    ]
    traj_files = []
    for idx, (chunk_path, start_frame) in enumerate(chunk_paths):
        arg = (chunk_path, start_frame, args.interval, args.checkpoints_dir, idx, 0)  # always use GPU 0
        traj_files.append(process_chunk(arg))

    # Step 3: Merge trajectories
    merge_trajectories(traj_files, args.output_traj)

if __name__ == '__main__':
    main() 