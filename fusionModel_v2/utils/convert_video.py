import cv2
import argparse
from pathlib import Path


def convert_video(input_path, output_path, fps=1, height=1080):
    """
    Convert video to specified fps and resolution (1080p)

    Args:
        input_path: Input video file path
        output_path: Output video file path
        fps: Target frames per second (default: 1)
        height: Target height in pixels (default: 1080)
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_path}")
        return False

    # Get input video properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {orig_width}x{orig_height} @ {orig_fps}fps, {total_frames} frames")

    # Calculate output width to maintain aspect ratio
    width = int(orig_width * height / orig_height)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Cannot create output video {output_path}")
        cap.release()
        return False

    # Calculate frame skip ratio
    frame_skip = int(orig_fps / fps)
    frame_count = 0
    written_frames = 0

    print(f"Output: {width}x{height} @ {fps}fps")
    print("Processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Write every nth frame
        if frame_count % frame_skip == 0:
            resized = cv2.resize(frame, (width, height))
            out.write(resized)
            written_frames += 1

        frame_count += 1

    cap.release()
    out.release()

    print(f"✅ Done! Written {written_frames} frames to {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to 1fps and 1080p")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("-o", "--output", help="Output video file path (default: input_1fps_1080p.mp4)")
    parser.add_argument("--fps", type=int, default=1, help="Target FPS (default: 1)")
    parser.add_argument("--height", type=int, default=1080, help="Target height (default: 1080)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        exit(1)

    if args.output:
        output_path = args.output
    else:
        output_path = input_path.stem + "_1fps_1080p.mp4"

    convert_video(str(input_path), output_path, fps=args.fps, height=args.height)
