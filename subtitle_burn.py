#!/usr/bin/env python3
"""
Auto-subtitle burner: extract audio -> transcribe (faster-whisper, word timestamps)
-> generate TikTok-style word-by-word captions -> burn into the video with ffmpeg.

Usage:
    python subtitle_burn.py input.mp4 output.mp4
    python subtitle_burn.py input.mp4 output.mp4 --model_size small --device cpu

Requires ffmpeg on PATH and the `faster-whisper` Python package.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ffmpeg not found on PATH.\n"
            "Windows: download a build from https://www.gyan.dev/ffmpeg/builds/, "
            "unzip it and add the 'bin' folder to your PATH (or pass --ffmpeg-path).\n"
            "macOS: brew install ffmpeg\n"
            "Linux: sudo apt install ffmpeg"
        )


def extract_audio(video_path: str, audio_path: str, ffmpeg_bin: str):
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        audio_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        sys.exit(f"ffmpeg failed to extract audio:\n{result.stderr.decode(errors='ignore')}")


def transcribe(
    audio_path: str,
    model_size: str,
    device: str,
    compute_type: str,
    language: str | None,
):
    if WhisperModel is None:
        sys.exit(
            "faster-whisper is not installed. Install it with:\n"
            "    pip install faster-whisper"
        )

    print(f"Loading whisper model '{model_size}' (device={device}, compute_type={compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        language=language,
    )
    print(f"Detected language: {info.language} (p={info.language_probability:.2f})")

    words = []
    for segment in segments:
        if not segment.words:
            continue
        for word in segment.words:
            text = word.word.strip()
            if text:
                words.append((word.start, word.end, text))

    if not words:
        sys.exit("No speech detected in the audio track.")
    return words


def ass_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    cs = round(seconds * 100)
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTok,{font},{font_size},{font_color},&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,{outline},0,{alignment},40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

FONT_COLOR_PRESETS = {
    "white": "FFFFFF",
    "yellow": "FFFF00",
    "red": "FF3B30",
    "green": "34C759",
    "cyan": "00FFFF",
}


def resolve_font_color(value: str) -> str:
    """Accepts a preset name or a #RRGGBB / RRGGBB hex string, returns an
    ASS PrimaryColour value (format is &HAABBGGRR, i.e. blue/green/red
    order - the reverse of normal hex - with 00 alpha for fully opaque)."""
    hex_rgb = FONT_COLOR_PRESETS.get(value.lower(), value.lstrip("#"))
    if len(hex_rgb) != 6:
        sys.exit(f"Invalid --font_color: {value!r}. Use a preset ({', '.join(FONT_COLOR_PRESETS)}) or a #RRGGBB hex code.")
    try:
        r, g, b = hex_rgb[0:2], hex_rgb[2:4], hex_rgb[4:6]
        int(hex_rgb, 16)
    except ValueError:
        sys.exit(f"Invalid --font_color: {value!r}. Use a preset ({', '.join(FONT_COLOR_PRESETS)}) or a #RRGGBB hex code.")
    return f"&H00{b}{g}{r}"


def build_ass(
    words,
    ass_path: str,
    res_x: int,
    res_y: int,
    font: str,
    font_size: int,
    font_color: str,
    outline: int,
    alignment: int,
    margin_v: int,
    min_word_duration: float,
    uppercase: bool,
):
    header = ASS_HEADER.format(
        res_x=res_x,
        res_y=res_y,
        font=font,
        font_size=font_size,
        font_color=resolve_font_color(font_color),
        outline=outline,
        alignment=alignment,
        margin_v=margin_v,
    )
    lines = [header]
    for start, end, text in words:
        if uppercase:
            text = text.upper()
        text = text.replace("{", "(").replace("}", ")")
        if end - start < min_word_duration:
            end = start + min_word_duration
        lines.append(
            f"Dialogue: 0,{ass_timestamp(start)},{ass_timestamp(end)},TikTok,,0,0,0,,{text}\n"
        )
    with open(ass_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def probe_resolution(video_path: str, ffprobe_bin: str):
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        video_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        sys.exit(f"ffprobe failed to read video resolution:\n{result.stderr.decode(errors='ignore')}")
    width, height = result.stdout.decode().strip().split("x")
    return int(width), int(height)


def escape_filter_path(path: str) -> str:
    # ffmpeg's filtergraph parser splits filter args on ":" and unescapes "\\"
    # itself before that split happens, so a Windows drive-letter colon needs
    # a double backslash (one level to survive the unescape, one to escape
    # the colon) - a single "\:" is not enough and gets misparsed as a
    # second positional option (e.g. "original_size").
    return path.replace("\\", "/").replace(":", "\\\\:")


def burn_subtitles(video_path: str, ass_path: str, output_path: str, ffmpeg_bin: str, fontsdir: str | None):
    filter_arg = f"ass={escape_filter_path(ass_path)}"
    if fontsdir:
        filter_arg += f":fontsdir={escape_filter_path(fontsdir)}"

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
        sys.exit(f"ffmpeg failed to burn subtitles:\n{result.stderr.decode(errors='ignore')}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the input video file")
    parser.add_argument("output", help="Path to write the subtitled output video")
    parser.add_argument("--model_size", default="small",
                         help="faster-whisper model size (tiny/base/small/medium/large-v3/...). "
                              "Default 'small': best accuracy/speed balance on CPU-only machines.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                         help="Inference device. Use 'cuda' only if you have an NVIDIA GPU + CUDA installed.")
    parser.add_argument("--compute_type", default="int8",
                         help="faster-whisper compute type, e.g. int8 (fast on CPU), float16 (GPU).")
    parser.add_argument("--language", default=None, help="Force a language code (e.g. 'id', 'en'); default: auto-detect")
    parser.add_argument("--font", default="Arial Black", help="Font family for the burned-in captions")
    parser.add_argument("--font_size", type=int, default=18, help="Base font size as a fraction of video height (see --font_size_pct)")
    parser.add_argument("--font_size_pct", type=float, default=0.075,
                         help="Font size as a fraction of video height (default 7.5%%, a typical TikTok caption size)")
    parser.add_argument("--font_color", default="yellow",
                         help=f"Caption color: a preset ({', '.join(FONT_COLOR_PRESETS)}) or a #RRGGBB hex code")
    parser.add_argument("--outline", type=int, default=4, help="Outline thickness in pixels-ish (ASS units)")
    parser.add_argument("--alignment", type=int, default=5,
                         help="ASS alignment (numpad layout): 2=bottom-center, 5=middle-center, 8=top-center")
    parser.add_argument("--margin_v", type=int, default=120, help="Vertical margin from the aligned edge")
    parser.add_argument("--min_word_duration", type=float, default=0.12,
                         help="Minimum seconds each word stays on screen, so very short words are still readable")
    parser.add_argument("--no_uppercase", action="store_true", help="Keep original casing instead of forcing UPPERCASE captions")
    parser.add_argument("--fontsdir", default=None, help="Directory containing custom font files (.ttf/.otf) if --font isn't a system font")
    parser.add_argument("--ffmpeg-path", default="ffmpeg", help="Path to the ffmpeg executable")
    parser.add_argument("--ffprobe-path", default="ffprobe", help="Path to the ffprobe executable")
    parser.add_argument("--keep-ass", default=None, help="Optional path to also save the generated .ass subtitle file")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Input file not found: {args.input}")

    if args.ffmpeg_path == "ffmpeg" or args.ffprobe_path == "ffprobe":
        check_ffmpeg()

    with tempfile.TemporaryDirectory(prefix="subtitle_burn_") as tmp_dir:
        audio_path = os.path.join(tmp_dir, "audio.wav")
        ass_path = os.path.join(tmp_dir, "subtitle.ass")

        print(f"[1/4] Extracting audio from {args.input} ...")
        extract_audio(args.input, audio_path, args.ffmpeg_path)

        print("[2/4] Transcribing audio (this can take a while on CPU)...")
        words = transcribe(audio_path, args.model_size, args.device, args.compute_type, args.language)
        print(f"    -> {len(words)} words transcribed")

        print("[3/4] Building TikTok-style subtitle track...")
        width, height = probe_resolution(args.input, args.ffprobe_path)
        font_size = max(10, round(height * args.font_size_pct))
        build_ass(
            words,
            ass_path,
            res_x=width,
            res_y=height,
            font=args.font,
            font_size=font_size,
            font_color=args.font_color,
            outline=args.outline,
            alignment=args.alignment,
            margin_v=args.margin_v,
            min_word_duration=args.min_word_duration,
            uppercase=not args.no_uppercase,
        )
        if args.keep_ass:
            shutil.copyfile(ass_path, args.keep_ass)
            print(f"    -> subtitle file saved to {args.keep_ass}")

        print(f"[4/4] Burning subtitles into {args.output} ...")
        burn_subtitles(args.input, ass_path, args.output, args.ffmpeg_path, args.fontsdir)

    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
