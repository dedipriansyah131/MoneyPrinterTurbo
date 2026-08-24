#!/usr/bin/env python3
"""
Clip extractor: from a long video, detect silence vs speech (ffmpeg
silencedetect + faster-whisper word timestamps), group speech into
15-60s candidate clips, and cut each one out with ffmpeg.

Clip boundaries always land on a detected silence gap (never mid-speech,
except when a single uninterrupted speech run is itself longer than
--max_clip, which gets hard-trimmed). Each clip gets a companion .txt
with its transcript, plus a manifest.json summarizing all clips - so you
can skim the text and pick which ones are worth using before opening any
video.

Usage:
    python clip_extractor.py long_video.mp4 clips_out/
    python clip_extractor.py long_video.mp4 clips_out/ --min_clip 15 --max_clip 60

Requires ffmpeg on PATH and faster-whisper:
    pip install faster-whisper
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ffmpeg not found on PATH.\n"
            "Windows: download a build from https://www.gyan.dev/ffmpeg/builds/, "
            "unzip it and add the 'bin' folder to your PATH.\n"
            "macOS: brew install ffmpeg\n"
            "Linux: sudo apt install ffmpeg"
        )


def probe_duration(video_path: str, ffprobe_bin: str) -> float:
    cmd = [
        ffprobe_bin, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        sys.exit(f"ffprobe failed to read video duration:\n{result.stderr.decode(errors='ignore')}")
    return float(result.stdout.decode().strip())


def extract_audio(video_path: str, audio_path: str, ffmpeg_bin: str):
    cmd = [
        ffmpeg_bin, "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        audio_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        sys.exit(f"ffmpeg failed to extract audio:\n{result.stderr.decode(errors='ignore')}")


def transcribe_words(audio_path: str, model_size: str, device: str, compute_type: str, language):
    if WhisperModel is None:
        sys.exit(
            "faster-whisper is not installed. Install it with:\n"
            "    pip install faster-whisper"
        )
    print(f"Loading whisper model '{model_size}' (device={device}, compute_type={compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        audio_path, word_timestamps=True, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500), language=language,
    )
    print(f"Detected language: {info.language} (p={info.language_probability:.2f})")

    words = []
    for segment in segments:
        for word in segment.words or []:
            text = word.word.strip()
            if text:
                words.append((word.start, word.end, text))
    return words


def detect_silences(audio_path: str, ffmpeg_bin: str, silence_db: float, min_silence: float, duration: float):
    cmd = [
        ffmpeg_bin, "-i", audio_path,
        "-af", f"silencedetect=noise={silence_db}dB:d={min_silence}",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr = result.stderr.decode(errors="ignore")

    silences = []
    pending_start = None
    for line in stderr.splitlines():
        m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        if m:
            pending_start = float(m.group(1))
            continue
        m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
        if m and pending_start is not None:
            silences.append((pending_start, float(m.group(1))))
            pending_start = None
    if pending_start is not None:
        silences.append((pending_start, duration))
    return silences


def speech_intervals_from_silences(silences, duration: float):
    speech = []
    cursor = 0.0
    for start, end in sorted(silences):
        if start > cursor:
            speech.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        speech.append((cursor, duration))
    return speech


def build_clips(speech_intervals, min_dur: float, max_dur: float):
    clips = []
    i = 0
    n = len(speech_intervals)
    while i < n:
        clip_start = speech_intervals[i][0]
        clip_end = speech_intervals[i][1]
        j = i
        while j + 1 < n and (speech_intervals[j + 1][1] - clip_start) <= max_dur:
            j += 1
            clip_end = speech_intervals[j][1]

        if clip_end - clip_start > max_dur:
            # a single uninterrupted speech run longer than max_dur: hard trim
            clip_end = clip_start + max_dur

        clips.append((clip_start, clip_end))
        i = j + 1

    return [(s, e) for s, e in clips if e - s >= 1.0]


def words_text_in_range(words, start: float, end: float) -> str:
    return " ".join(w for ws, we, w in words if ws >= start - 0.05 and we <= end + 0.05)


def cut_clip(video_path: str, start: float, end: float, output_path: str, ffmpeg_bin: str, reencode: bool):
    duration = end - start
    if reencode:
        cmd = [
            ffmpeg_bin, "-y", "-ss", f"{start:.3f}", "-i", video_path, "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac",
            output_path,
        ]
    else:
        cmd = [
            ffmpeg_bin, "-y", "-ss", f"{start:.3f}", "-i", video_path, "-t", f"{duration:.3f}",
            "-c", "copy", "-avoid_negative_ts", "make_zero",
            output_path,
        ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        sys.exit(f"ffmpeg failed to cut clip {output_path}:\n{result.stderr.decode(errors='ignore')}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the long input video file")
    parser.add_argument("output_dir", help="Directory to write candidate clips + manifest into")
    parser.add_argument("--min_clip", type=float, default=15.0, help="Minimum clip duration in seconds")
    parser.add_argument("--max_clip", type=float, default=60.0, help="Maximum clip duration in seconds")
    parser.add_argument("--min_silence", type=float, default=0.6,
                         help="Minimum silence duration (seconds) to count as a pause/cut point")
    parser.add_argument("--silence_db", type=float, default=-30.0,
                         help="Silence threshold in dBFS - lower (more negative) is stricter/quieter")
    parser.add_argument("--model_size", default="small", help="faster-whisper model size")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--compute_type", default="int8")
    parser.add_argument("--language", default=None, help="Force a language code; default: auto-detect")
    parser.add_argument("--reencode", action="store_true",
                         help="Re-encode clips for frame-accurate cuts (slower than the default stream copy)")
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--ffprobe-path", default="ffprobe")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Input file not found: {args.input}")
    if args.ffmpeg_path == "ffmpeg" or args.ffprobe_path == "ffprobe":
        check_ffmpeg()

    os.makedirs(args.output_dir, exist_ok=True)
    audio_path = os.path.join(args.output_dir, "_full_audio.wav")

    print("[1/5] Probing video ...")
    duration = probe_duration(args.input, args.ffprobe_path)
    print(f"    -> duration: {duration:.1f}s")

    print("[2/5] Extracting audio ...")
    extract_audio(args.input, audio_path, args.ffmpeg_path)

    print("[3/5] Transcribing (for clip text previews) ...")
    words = transcribe_words(audio_path, args.model_size, args.device, args.compute_type, args.language)
    print(f"    -> {len(words)} words transcribed")

    print(f"[4/5] Detecting silence (threshold {args.silence_db}dB, min {args.min_silence}s) ...")
    silences = detect_silences(audio_path, args.ffmpeg_path, args.silence_db, args.min_silence, duration)
    speech_intervals = speech_intervals_from_silences(silences, duration)
    print(f"    -> {len(silences)} silence gaps, {len(speech_intervals)} speech segments")

    os.remove(audio_path)

    clips = build_clips(speech_intervals, args.min_clip, args.max_clip)
    if not clips:
        sys.exit("No candidate clips found - the video may be silent, or --min_silence/--silence_db need tuning.")
    print(f"    -> {len(clips)} candidate clips (target {args.min_clip:.0f}-{args.max_clip:.0f}s each)")

    print("[5/5] Cutting clips ...")
    manifest = []
    for idx, (start, end) in enumerate(clips, start=1):
        clip_name = f"clip_{idx:03d}.mp4"
        clip_path = os.path.join(args.output_dir, clip_name)
        text = words_text_in_range(words, start, end)

        cut_clip(args.input, start, end, clip_path, args.ffmpeg_path, args.reencode)

        txt_path = os.path.join(args.output_dir, f"clip_{idx:03d}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        manifest.append({
            "file": clip_name,
            "start": round(start, 2),
            "end": round(end, 2),
            "duration": round(end - start, 2),
            "text": text,
        })
        preview = (text[:70] + "...") if len(text) > 70 else text
        print(f"    -> {clip_name} [{start:.1f}s - {end:.1f}s, {end - start:.1f}s]: {preview!r}")

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done: {len(clips)} clips written to {args.output_dir}")


if __name__ == "__main__":
    main()
