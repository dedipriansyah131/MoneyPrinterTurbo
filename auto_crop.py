#!/usr/bin/env python3
"""
Auto-crop to 9:16: sample the video once per second, detect the largest
face with OpenCV's Haar cascade (frontal + profile), then crop the video
with ffmpeg so the crop window pans smoothly to follow the face's
position over time.

This is a lightweight "sample, then interpolate" tracker, not full
per-frame face tracking - face position is only detected once per
--sample_interval seconds, but the crop pans continuously between those
samples instead of jumping.

Usage:
    python auto_crop.py input.mp4 output.mp4
    python auto_crop.py input.mp4 output.mp4 --sample_interval 1.0 --out_width 1080 --out_height 1920

Requires ffmpeg on PATH and opencv-python-headless (with bundled Haar
cascades - pin to 4.10.0.84 if a newer release drops them):
    pip install opencv-python-headless==4.10.0.84
"""
import argparse
import os
import shutil
import subprocess
import sys

try:
    import cv2
except ImportError:
    cv2 = None


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ffmpeg not found on PATH.\n"
            "Windows: download a build from https://www.gyan.dev/ffmpeg/builds/, "
            "unzip it and add the 'bin' folder to your PATH.\n"
            "macOS: brew install ffmpeg\n"
            "Linux: sudo apt install ffmpeg"
        )


def load_face_cascades(cascade_path: str):
    if cv2 is None:
        sys.exit(
            "opencv-python is not installed. Install it with:\n"
            "    pip install opencv-python-headless==4.10.0.84\n"
            "(pinned: newer 5.x releases have dropped the bundled Haar cascade files)"
        )

    def load(name):
        cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, name))
        return cascade if not cascade.empty() else None

    if cascade_path:
        if not os.path.isfile(cascade_path):
            sys.exit(f"Face cascade file not found: {cascade_path}")
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            sys.exit(f"Failed to load face cascade from: {cascade_path}")
        return [cascade]

    # Try frontal faces first (most reliable), then profile faces so a
    # turned/tilted head is still caught. Newer opencv-python releases have
    # dropped the bundled cascade files entirely (see the pin above), so
    # missing files here mean the install needs to be fixed, not a bug.
    cascades = [
        load("haarcascade_frontalface_default.xml"),
        load("haarcascade_frontalface_alt2.xml"),
        load("haarcascade_profileface.xml"),
    ]
    cascades = [c for c in cascades if c is not None]
    if not cascades:
        sys.exit(
            "No Haar cascade files found bundled with opencv-python. Try:\n"
            "    pip install opencv-python-headless==4.10.0.84\n"
            "or pass --cascade_path pointing at a haarcascade_frontalface_default.xml file."
        )
    return cascades


def probe_video(video_path: str, ffprobe_bin: str):
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1",
        video_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        sys.exit(f"ffprobe failed to read video info:\n{result.stderr.decode(errors='ignore')}")

    info = {}
    for line in result.stdout.decode().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key] = value
    return int(info["width"]), int(info["height"]), float(info["duration"])


def detect_largest_face(gray, cascades, min_face_size: int):
    min_size = (min_face_size, min_face_size)

    for cascade in cascades:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
        if len(faces):
            return max(faces, key=lambda f: f[2] * f[3])

    # Profile cascades are trained on one facing direction; flip the frame
    # to also catch faces turned the other way, then mirror the box back.
    if cascades:
        flipped = cv2.flip(gray, 1)
        faces = cascades[-1].detectMultiScale(flipped, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return (gray.shape[1] - x - w, y, w, h)

    return None


def detect_face_centers(video_path: str, duration: float, sample_interval: float, axis: str, cascades, min_face_size: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"OpenCV failed to open video: {video_path}")

    samples = []  # list of (time, center_or_None)
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            samples.append((t, None))
            t += sample_interval
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = detect_largest_face(gray, cascades, min_face_size)
        if face is None:
            samples.append((t, None))
        else:
            x, y, w, h = face
            center = (x + w / 2) if axis == "x" else (y + h / 2)
            samples.append((t, center))
        t += sample_interval

    cap.release()
    return samples


def fill_and_smooth(samples, default_center: float, smooth_window: int):
    values = [c for _, c in samples]
    n = len(values)

    last = None
    for i in range(n):
        if values[i] is None:
            values[i] = last
        else:
            last = values[i]

    nxt = None
    for i in range(n - 1, -1, -1):
        if values[i] is None:
            values[i] = nxt
        else:
            nxt = values[i]

    values = [v if v is not None else default_center for v in values]

    half = smooth_window // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = values[lo:hi]
        smoothed.append(sum(window) / len(window))
    return smoothed


def build_position_expr(times, positions) -> str:
    """Piecewise-linear interpolation between samples, as an ffmpeg eval
    expression using the filter's built-in 't' (current timestamp)
    variable. This makes the crop pan continuously instead of jumping to a
    new position once per sample."""
    expr = f"{positions[-1]:.2f}"
    for i in range(len(positions) - 2, -1, -1):
        t0, t1 = times[i], times[i + 1]
        p0, p1 = positions[i], positions[i + 1]
        if t1 <= t0:
            continue
        segment = f"({p0:.2f}+({p1:.2f}-({p0:.2f}))*(t-{t0:.3f})/{(t1 - t0):.3f})"
        expr = f"if(lt(t,{t1:.3f}),{segment},{expr})"
    return expr


def crop_and_scale(
    video_path: str,
    output_path: str,
    crop_w: int,
    crop_h: int,
    x_expr: str,
    y_expr: str,
    out_width: int,
    out_height: int,
    ffmpeg_bin: str,
):
    filter_arg = (
        f"crop=w={crop_w}:h={crop_h}:x='{x_expr}':y='{y_expr}':exact=1,"
        f"scale={out_width}:{out_height}"
    )
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", video_path,
        "-vf", filter_arg,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        sys.exit(f"ffmpeg failed to crop/scale video:\n{result.stderr.decode(errors='ignore')}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("output", help="Path to write the cropped 9:16 output video")
    parser.add_argument("--sample_interval", type=float, default=0.5,
                         help="Seconds between face-detection samples (default: 0.5 - lower reacts faster to movement "
                              "but takes longer to process and samples more false positives)")
    parser.add_argument("--out_width", type=int, default=1080, help="Output video width")
    parser.add_argument("--out_height", type=int, default=1920, help="Output video height")
    parser.add_argument("--min_face_size", type=int, default=60,
                         help="Minimum face size in pixels for detection (smaller = more false positives)")
    parser.add_argument("--smooth_window", type=int, default=3,
                         help="Moving-average window (in samples) to reduce jitter between detections")
    parser.add_argument("--cascade_path", default=None,
                         help="Path to a haarcascade_frontalface_default.xml file (default: bundled with opencv-python)")
    parser.add_argument("--ffmpeg-path", default="ffmpeg", help="Path to the ffmpeg executable")
    parser.add_argument("--ffprobe-path", default="ffprobe", help="Path to the ffprobe executable")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Input file not found: {args.input}")

    if args.ffmpeg_path == "ffmpeg" or args.ffprobe_path == "ffprobe":
        check_ffmpeg()

    cascades = load_face_cascades(args.cascade_path)

    width, height, duration = probe_video(args.input, args.ffprobe_path)
    target_ratio = 9 / 16

    if width / height > target_ratio:
        # Wider than target: keep full height, pan horizontally.
        axis = "x"
        crop_h = height
        crop_w = round(height * target_ratio)
        pan_min, pan_max = 0, width - crop_w
        default_center = width / 2
    else:
        # Taller/narrower than target: keep full width, pan vertically.
        axis = "y"
        crop_w = width
        crop_h = round(width / target_ratio)
        pan_min, pan_max = 0, height - crop_h
        default_center = height / 2

    print(f"[1/4] Source: {width}x{height}, {duration:.1f}s. Crop window: {crop_w}x{crop_h}, panning on '{axis}'.")

    print(f"[2/4] Sampling faces every {args.sample_interval}s ...")
    samples = detect_face_centers(args.input, duration, args.sample_interval, axis, cascades, args.min_face_size)
    detected = sum(1 for _, c in samples if c is not None)
    print(f"    -> {detected}/{len(samples)} samples had a detected face")
    if detected == 0:
        print("    -> no faces detected anywhere, falling back to a centered crop")

    print("[3/4] Building crop position track (smooth pan, not per-second jumps) ...")
    crop_size = crop_w if axis == "x" else crop_h
    centers = fill_and_smooth(samples, default_center, args.smooth_window)
    positions = [min(max(c - crop_size / 2, pan_min), pan_max) for c in centers]
    times = [t for t, _ in samples]

    pos_expr = build_position_expr(times, positions)
    x_expr, y_expr = (pos_expr, "0") if axis == "x" else ("0", pos_expr)

    print(f"[4/4] Cropping and scaling to {args.out_width}x{args.out_height} ...")
    crop_and_scale(
        args.input, args.output, crop_w, crop_h, x_expr, y_expr,
        args.out_width, args.out_height, args.ffmpeg_path,
    )

    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
