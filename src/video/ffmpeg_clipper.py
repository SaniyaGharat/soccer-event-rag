"""
Ffmpeg Video Clipper Module

Extracts sub-clips from match video files based on start and end timestamp ranges.
Uses imageio-ffmpeg or system ffmpeg to ensure H.264 HTML5 video browser compatibility.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional

try:
    import imageio_ffmpeg
    IMAGEIO_FFMPEG_AVAILABLE = True
except ImportError:
    IMAGEIO_FFMPEG_AVAILABLE = False


def get_ffmpeg_binary_path() -> Optional[str]:
    """Locates system ffmpeg or bundled imageio-ffmpeg executable."""
    sys_path = shutil.which("ffmpeg")
    if sys_path:
        return sys_path
    if IMAGEIO_FFMPEG_AVAILABLE:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return None


class VideoClipper:
    """
    Sub-clip extractor using ffmpeg.
    """

    def __init__(self, output_dir: str = "data/clips"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ffmpeg_exe = get_ffmpeg_binary_path()

    def extract_clip(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
        clip_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Extracts sub-clip between start_sec and end_sec.
        Returns path to extracted MP4 clip file encoded with H.264 for HTML5 web playback.
        """
        src = Path(video_path)
        if not src.exists():
            print(f"[!] Source video file not found at: {src}")
            return None

        duration = max(1.0, end_sec - start_sec)
        out_filename = clip_name or f"clip_{int(start_sec)}_{int(end_sec)}.mp4"
        out_path = self.output_dir / out_filename

        # If clip already exists and is valid, return immediately to avoid locking open files
        if out_path.exists() and out_path.stat().st_size > 5000:
            return str(out_path)

        if self.ffmpeg_exe:
            # Two-stage frame-accurate seeking:
            # 1. Fast seek to ~1 second before target
            # 2. Precise decode from the relative offset to exact frame
            # 3. -movflags +faststart for instantaneous HTML5 video browser playback
            pre_seek = max(0.0, start_sec - 1.0)
            rel_seek = start_sec - pre_seek

            cmd = [
                self.ffmpeg_exe,
                "-y",  # Overwrite output
                "-ss", f"{pre_seek:.3f}",
                "-i", str(src),
                "-ss", f"{rel_seek:.3f}",
                "-t", f"{duration:.3f}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                str(out_path)
            ]
            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=25)
                if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
                    return str(out_path)
                else:
                    print(f"[!] Ffmpeg process error: {res.stderr}")
            except Exception as e:
                print(f"[!] Ffmpeg clip extraction failed: {e}")

        # Fallback: Return original video path if clip cutting unavailable
        return str(src)
