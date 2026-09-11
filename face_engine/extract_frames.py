import cv2  # type: ignore
import os

def extract_frames(video_path, output_dir):
    """Extracts all frames from a video and saves them as images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    while success:
        # Save frame as JPEG file, prefixed with video name
        frame_name = f"{video_basename}_frame_{count:04d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, image)
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    print(f"Extracted {count} frames from {os.path.basename(video_path)} to {output_dir}")

def process_all_videos(videos_dir):
    """Finds all videos in a directory, extracts frames, and saves them in a single folder."""
    output_dir = os.path.join(videos_dir, "all_frames")
    
    # List all files in the directory
    for filename in os.listdir(videos_dir):
        if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".mov"):
            video_path = os.path.join(videos_dir, filename)
            
            print(f"Processing {filename}...")
            extract_frames(video_path, output_dir)
            
if __name__ == "__main__":
    # Define the directory containing the videos
    target_directory = r"c:\projects\temp_magikCLick\captured_videos"
    
    if os.path.exists(target_directory):
        print(f"Starting frame extraction for videos in {target_directory}...")
        process_all_videos(target_directory)
        print("Done!")
    else:
        print(f"Directory {target_directory} not found.")
