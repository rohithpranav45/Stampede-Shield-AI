import open3d as o3d
import glob
import time
import os
import numpy as np
import imageio  # For GIF creation

def animate_point_clouds(checkpoints_dir, delay=0.2, output_gif="animation.gif"):
    ply_files = sorted(glob.glob(os.path.join(checkpoints_dir, "*.ply")))
    if not ply_files:
        print("No PLY files found in", checkpoints_dir)
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Crowd Animation", width=960, height=720, visible=True)
    
    # Set black background and white points
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0, 0, 0])  # Black background
    render_opt.point_size = 3.0  # Larger points for better visibility
    
    pcd = o3d.geometry.PointCloud()
    added = False
    frames = []  # To store animation frames

    for ply_file in ply_files:
        # Load and color points white
        pcd.points = o3d.io.read_point_cloud(ply_file).points
        pcd.paint_uniform_color([1, 1, 1])  # White points
        
        if not added:
            vis.add_geometry(pcd)
            added = True
        else:
            vis.update_geometry(pcd)
        
        vis.poll_events()
        vis.update_renderer()
        
        # Capture frame
        image = vis.capture_screen_float_buffer()
        frames.append((np.asarray(image) * 255).astype(np.uint8))
        
        print(f"Processed {os.path.basename(ply_file)}")
        time.sleep(delay)

    # Save as GIF
    imageio.mimsave(output_gif, frames, fps=1/delay, loop=0)
    print(f"Animation saved as {output_gif}")
    
    vis.destroy_window()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Animate 3D crowd checkpoints as a fluid animation.")
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Directory with checkpoint PLY files')
    parser.add_argument('--delay', type=float, default=0.2, help='Delay (seconds) between frames')
    parser.add_argument('--output_gif', type=str, default='animation.gif', help='Output GIF filename')
    args = parser.parse_args()

    animate_point_clouds(args.checkpoints_dir, delay=args.delay, output_gif=args.output_gif)