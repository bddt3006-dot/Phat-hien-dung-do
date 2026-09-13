import cv2
import os
import glob

def create_video_from_frames(frames_dir, output_video_path, fps=25):
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    images = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if not images:
        print(f"Error: No images found in {frames_dir}")
        return
    
    first_frame = cv2.imread(images[0])
    h, w, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    print(f"Compiling {len(images)} frames into {output_video_path} at {fps} FPS ({w}x{h})...")
    for img_path in images:
        frame = cv2.imread(img_path)
        out.write(frame)
        
    out.release()
    print("Sample video successfully created at:", output_video_path)

if __name__ == "__main__":
    seq_dir = os.path.join("Data UA Detrac", "DETRAC-Images", "DETRAC-Images", "MVI_20011")
    output_path = os.path.join("data", "sample.mp4")
    create_video_from_frames(seq_dir, output_path)
