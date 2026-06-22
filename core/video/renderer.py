"""
Video Renderer — Novel Video Factory v4

Assembles images + audio into 10-minute clips, then stitches into final video.

BUG FIXES vs v3:
- torch imported inside functions (not at top-level) to avoid crash on CPU-only
- FFmpeg concat list uses ABSOLUTE paths to fix 'file not found' errors
- Ken Burns effect wrapped in try/except so a bad image doesn't kill the clip
- subtitle rendering uses DejaVu font (always available on Kaggle/Linux)
- clip.close() called in finally block to prevent resource leaks

v5: Configurable presentation layout (blurred_background / bordered_panel),
configurable output aspect ratio decoupled from native generation
resolution, real pan_left/pan_right motion (previously mapped to zoom for
"safety"), and dual audio-mode support: continuous per-clip narration with
per-scene (audio_start, audio_end) windows, OR legacy per-scene .wav files.
"""
import gc
import json
import logging
import os
import random
import subprocess
import hashlib
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _parse_aspect_ratio(ratio_str: str, long_edge: int) -> Tuple[int, int]:
    """
    Parse a "W:H" aspect ratio string into concrete (width, height) pixel
    dimensions, sized so the longer logical edge equals `long_edge`.
    Rounds to even numbers (libx264 requires even width/height).
    """
    try:
        rw, rh = ratio_str.split(":")
        rw, rh = float(rw), float(rh)
    except Exception:
        rw, rh = 3.0, 4.0  # fallback: portrait 3:4

    def _even(n):
        n = int(round(n))
        return n - (n % 2)

    if rw >= rh:
        w = long_edge
        h = long_edge * rh / rw
    else:
        h = long_edge
        w = long_edge * rw / rh
    return _even(w), _even(h)


class VideoRenderer:
    """
    Assembles per-scene images and audio into MP4 clips.
    Each clip is ~10 minutes. Then stitches all clips into a master video.
    """
    def __init__(self, project_dir: str, config: dict = None):
        self.project_dir = project_dir
        self.config = config or {}
        vid_cfg = self.config.get("video", {})
        self.fps = vid_cfg.get("fps", 24)
        self.font = vid_cfg.get("font", "DejaVu-Sans-Bold")
        self.font_size = vid_cfg.get("font_size", 40)
        self.ken_burns_speed = vid_cfg.get("ken_burns_speed", 0.03)

        self.presentation_style = vid_cfg.get("presentation_style", "blurred_background")
        self.canvas_w, self.canvas_h = _parse_aspect_ratio(
            vid_cfg.get("output_aspect_ratio", "3:4"),
            vid_cfg.get("output_long_edge", 1440),
        )
        self.blur_radius = vid_cfg.get("blur_radius", 40)
        self.blur_brightness = vid_cfg.get("blur_brightness", 0.55)
        self.border_color = tuple(vid_cfg.get("border_color", [245, 245, 240]))
        self.border_thickness = vid_cfg.get("border_thickness", 3)

        self.output_dir = os.path.join(project_dir, "output")
        self.images_dir = os.path.join(self.output_dir, "images")
        self.audio_dir = os.path.join(self.output_dir, "audio")
        self.videos_dir = os.path.join(self.output_dir, "videos")
        self.clips_path = os.path.join(self.output_dir, "clips.json")
        self.final_video_path = os.path.join(self.videos_dir, "final_video.mp4")

        os.makedirs(self.videos_dir, exist_ok=True)
        logger.info(f"VideoRenderer: canvas={self.canvas_w}x{self.canvas_h} "
                    f"style={self.presentation_style}")

    def render(self):
        """Main entry point: renders all clips and stitches final video."""
        if not os.path.exists(self.clips_path):
            logger.error("clips.json not found — cannot render video")
            return

        with open(self.clips_path, "r", encoding="utf-8") as f:
            clips_data = json.load(f)

        logger.info(f"Rendering {len(clips_data)} clips…")
        rendered_clip_paths = []

        for clip in clips_data:
            clip_id = clip["clip_id"]
            clip_path = os.path.join(self.videos_dir, f"{clip_id}.mp4")
            
            # Use a hash of the clip's shots to detect changes
            clip_content_hash = hashlib.sha256(json.dumps(clip.get("shots", []), sort_keys=True).encode()).hexdigest()[:16]
            hash_file = clip_path + ".hash"
            
            existing_hash = ""
            if os.path.exists(hash_file):
                with open(hash_file, "r") as f:
                    existing_hash = f.read().strip()

            if os.path.exists(clip_path) and existing_hash == clip_content_hash:
                logger.info(f"Clip exists and unchanged, skipping: {clip_id}")
                rendered_clip_paths.append(clip_path)
                continue

            success = self._render_clip(clip, clip_path)
            if success:
                rendered_clip_paths.append(clip_path)
                with open(hash_file, "w") as f:
                    f.write(clip_content_hash)

        if not rendered_clip_paths:
            logger.error("No clips were rendered successfully")
            return

        self._stitch_final(rendered_clip_paths)
        self.generate_srt(clips_data)

    def generate_srt(self, clips_data: List[dict]):
        """Generate a master SRT file for the entire video."""
        srt_path = os.path.join(self.videos_dir, "subtitles.srt")
        logger.info(f"Generating SRT: {srt_path}")
        
        def format_time(seconds):
            hrs = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            msecs = int((seconds % 1) * 1000)
            return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"

        current_time = 0.0
        srt_index = 1
        
        try:
            from moviepy import AudioFileClip
            with open(srt_path, "w", encoding="utf-8") as f:
                for clip in clips_data:
                    narration_file = clip.get("narration_audio")
                    for shot in clip.get("shots", []):
                        text = shot.get("narration_text", "").strip()
                        if not text:
                            continue

                        if narration_file and shot.get("audio_start") is not None:
                            # Continuous-narration mode: timing already computed
                            # from real word-boundary timestamps during Stage 6.
                            start_str = format_time(shot["audio_start"])
                            end_str = format_time(shot["audio_end"])
                        else:
                            # Legacy per-scene audio file mode
                            sid = shot["scene_id"]
                            aud_path = os.path.join(self.audio_dir, f"{sid}.wav")
                            if not os.path.exists(aud_path):
                                continue
                            duration = AudioFileClip(aud_path).duration
                            start_str = format_time(current_time)
                            end_str = format_time(current_time + duration)
                            current_time += duration

                        f.write(f"{srt_index}\n{start_str} --> {end_str}\n{text}\n\n")
                        srt_index += 1
            logger.info("✓ SRT generated")
        except Exception as e:
            logger.error(f"Failed to generate SRT: {e}")

    def _render_clip(self, clip: dict, output_path: str) -> bool:
        """Render one 10-minute clip from its shots."""
        try:
            # Import moviepy 2.x style
            from moviepy import (ImageClip, AudioFileClip,
                                 concatenate_videoclips,
                                 TextClip, CompositeVideoClip, ColorClip)
        except ImportError:
            logger.error("moviepy not installed — cannot render video. "
                         "Run: pip install moviepy>=2.1.1")
            return False

        clip_id = clip["clip_id"]
        shots = clip.get("shots", [])
        logger.info(f"--- Rendering: {clip_id} ({len(shots)} shots) ---")

        # Continuous-narration mode: one shared audio track for the whole
        # clip, opened once here rather than per-shot.
        clip_audio = None
        narration_file = clip.get("narration_audio")
        if narration_file:
            narration_path = os.path.join(self.audio_dir, narration_file)
            if os.path.exists(narration_path):
                try:
                    clip_audio = AudioFileClip(narration_path)
                except Exception as e:
                    logger.warning(f"  Could not open continuous narration audio: {e}")
                    clip_audio = None

        shot_clips = []
        current_chapter = -1

        try:
            for shot in shots:
                # Layer: Chapter Intro Cards
                shot_chapter = shot.get("chapter", 1)
                if shot_chapter != current_chapter:
                    # New chapter detected — insert a title card
                    intro = self._render_chapter_intro(shot_chapter)
                    if intro:
                        shot_clips.append(intro)
                    current_chapter = shot_chapter

                sc = self._render_shot(shot, clip_id, clip_audio=clip_audio)
                if sc is not None:
                    shot_clips.append(sc)

            if not shot_clips:
                logger.warning(f"No valid shots in {clip_id}")
                return False

            try:
                final = concatenate_videoclips(shot_clips, method="compose")
                final.write_videofile(
                    output_path,
                    fps=self.fps,
                    codec="libx264",
                    audio_codec="aac",
                    logger=None,
                )
                logger.info(f"✓ Clip saved: {os.path.basename(output_path)}")
                return True
            except Exception as e:
                logger.error(f"Error writing {clip_id}: {e}")
                return False
            finally:
                for sc in shot_clips:
                    try:
                        sc.close()
                    except Exception:
                        pass
                gc.collect()
        finally:
            if clip_audio is not None:
                try:
                    clip_audio.close()
                except Exception:
                    pass

    def _render_shot(self, shot: dict, clip_id: str, clip_audio=None):
        """
        Render one shot: source panel composited onto the configured
        canvas (blurred-background or bordered-panel backdrop), animated
        with Ken Burns motion, captioned, and paired with its audio
        (a subclip of the shared continuous narration track if
        `clip_audio` is provided and the shot has audio_start/audio_end,
        otherwise its own legacy per-scene .wav file).
        """
        try:
            from moviepy import ImageClip, AudioFileClip, CompositeVideoClip
        except ImportError:
            return None

        shot_id = shot["scene_id"]
        img_path = os.path.join(self.images_dir, f"{shot_id}.png")
        if not os.path.exists(img_path):
            logger.warning(f"Missing image: {shot_id} — skipping")
            return None

        audio_clip = None
        owns_audio_clip = False
        try:
            if (clip_audio is not None and shot.get("audio_start") is not None
                    and shot.get("audio_end") is not None):
                start, end = shot["audio_start"], shot["audio_end"]
                if end <= start:
                    end = start + 1.0
                try:
                    audio_clip = clip_audio.subclipped(start, end)
                except AttributeError:
                    audio_clip = clip_audio.subclip(start, end)  # older moviepy
                duration = max(end - start, 0.3)
            else:
                aud_path = os.path.join(self.audio_dir, f"{shot_id}.wav")
                if not os.path.exists(aud_path):
                    logger.warning(f"Missing audio: {shot_id} — skipping")
                    return None
                audio_clip = AudioFileClip(aud_path)
                owns_audio_clip = True
                duration = max(audio_clip.duration, 1.0)

            try:
                from PIL import Image
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception as e:
                logger.error(f"Could not read image size for {shot_id}: {e}")
                return None

            backdrop_arr = self._build_backdrop(img_path, img_w, img_h)

            # ── Motion selection ────────────────────────────────────────
            camera = shot.get("camera_angle", "").lower()
            emotion = shot.get("emotion", "").lower()
            base_speed = self.ken_burns_speed

            if any(e in emotion for e in ["angry", "fighting", "shocked"]):
                speed_mult = 2.5
            elif "sad" in emotion or "fearful" in emotion:
                speed_mult = 0.8
            elif "close-up" in camera:
                speed_mult = 0.5
            elif any(c in camera for c in ["wide", "aerial"]):
                speed_mult = 1.5
            elif "low angle" in camera:
                speed_mult = 1.2
            else:
                speed_mult = 1.0
            speed = base_speed * speed_mult

            motion_type = shot.get("camera_motion")
            if motion_type not in {"zoom_in", "zoom_out", "pan_left", "pan_right", "static"}:
                # Fallback heuristic — used only when the scene planner
                # didn't supply an explicit camera_motion.
                if any(e in emotion for e in ["angry", "fighting", "shocked"]):
                    motion_type = "zoom_in"
                elif "sad" in emotion or "fearful" in emotion:
                    motion_type = "zoom_out"
                elif any(c in camera for c in ["wide", "aerial"]):
                    motion_type = random.choice(["pan_right", "pan_left"])
                elif "low angle" in camera or "close-up" in camera:
                    motion_type = "zoom_in"
                else:
                    motion_type = random.choice(["zoom_in", "zoom_out", "pan_right", "pan_left", "static"])

            base_scale = min(self.canvas_w / img_w, self.canvas_h / img_h)
            pan_margin = 0.15

            if motion_type in ("pan_left", "pan_right"):
                const_scale = base_scale * (1 + pan_margin)
                clip_w = img_w * const_scale
                clip_h = img_h * const_scale
                max_offset = max(0.0, (clip_w - self.canvas_w) / 2)

                def scale_func(t, _s=const_scale):
                    return _s

                def pos_func(t, _cw=clip_w, _ch=clip_h, _mo=max_offset, _dir=motion_type):
                    progress = (t / duration) if duration > 0 else 0.0
                    if _dir == "pan_right":
                        offset = _mo * (1 - 2 * progress)
                    else:  # pan_left
                        offset = -_mo * (1 - 2 * progress)
                    x = (self.canvas_w - _cw) / 2 + offset
                    y = (self.canvas_h - _ch) / 2
                    return (x, y)
            else:
                if motion_type == "zoom_in":
                    def scale_func(t, _b=base_scale, _sp=speed):
                        return _b * (1 + (_sp * t / duration if duration > 0 else 0))
                elif motion_type == "zoom_out":
                    def scale_func(t, _b=base_scale, _sp=speed):
                        return _b * (1 + _sp - (_sp * t / duration if duration > 0 else 0))
                else:  # static
                    def scale_func(t, _b=base_scale):
                        return _b

                def pos_func(t, _b=base_scale, _sf=scale_func):
                    s = _sf(t)
                    cw, ch = img_w * s, img_h * s
                    return ((self.canvas_w - cw) / 2, (self.canvas_h - ch) / 2)

            foreground = (ImageClip(img_path)
                          .with_duration(duration)
                          .resized(scale_func)
                          .with_position(pos_func))

            backdrop = ImageClip(backdrop_arr).with_duration(duration)

            composite = CompositeVideoClip(
                [backdrop, foreground], size=(self.canvas_w, self.canvas_h)
            )

            # Subtitles — drawn on the full canvas-sized composite so they
            # span the whole frame width and sit at the canvas bottom,
            # regardless of how wide the foreground panel itself is.
            subtitle_text = shot.get("narration_text", "").strip()
            if subtitle_text:
                try:
                    composite = self._add_subtitles(composite, subtitle_text, duration)
                except Exception as te:
                    logger.debug(f"Subtitle failed for {shot_id}: {te}")

            return composite.with_audio(audio_clip)

        except Exception as e:
            logger.error(f"Failed to render shot {shot_id}: {e}")
            if owns_audio_clip and audio_clip is not None:
                try:
                    audio_clip.close()
                except Exception:
                    pass
            return None

    def _build_backdrop(self, img_path: str, img_w: int, img_h: int):
        """
        Build the static canvas-sized backdrop layer as a numpy array,
        per the configured presentation_style:
        - blurred_background: the same panel, scaled to COVER the full
          canvas, blurred and darkened, so there are never black bars.
        - bordered_panel: a solid-color canvas with a thin border ring
          sized to the panel's base (no-zoom) footprint.
        """
        from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
        import numpy as np

        with Image.open(img_path) as im:
            src = im.convert("RGB").copy()

        if self.presentation_style == "bordered_panel":
            canvas = Image.new("RGB", (self.canvas_w, self.canvas_h), self.border_color)
            base_scale = min(self.canvas_w / img_w, self.canvas_h / img_h)
            fit_w, fit_h = int(img_w * base_scale), int(img_h * base_scale)
            x0 = (self.canvas_w - fit_w) // 2
            y0 = (self.canvas_h - fit_h) // 2
            t = self.border_thickness
            if t > 0:
                draw = ImageDraw.Draw(canvas)
                draw.rectangle(
                    [x0 - t, y0 - t, x0 + fit_w + t, y0 + fit_h + t],
                    fill=(20, 20, 20),
                )
            return np.array(canvas)

        # blurred_background (default)
        cover_scale = max(self.canvas_w / img_w, self.canvas_h / img_h)
        cov_w, cov_h = max(1, int(img_w * cover_scale)), max(1, int(img_h * cover_scale))
        bg = src.resize((cov_w, cov_h), Image.LANCZOS)
        # Center-crop to exactly canvas size
        left = (cov_w - self.canvas_w) // 2
        top = (cov_h - self.canvas_h) // 2
        bg = bg.crop((left, top, left + self.canvas_w, top + self.canvas_h))
        bg = bg.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        if self.blur_brightness != 1.0:
            bg = ImageEnhance.Brightness(bg).enhance(self.blur_brightness)
        return np.array(bg)

    def _render_chapter_intro(self, chapter_num: int):
        """Generate a cinematic black screen title card for a new chapter."""
        try:
            from moviepy import TextClip, CompositeVideoClip, ColorClip
            
            duration = 3.0
            w, h = self.canvas_w, self.canvas_h

            bg = ColorClip(size=(w, h), color=(0, 0, 0)).with_duration(duration)
            
            txt = TextClip(
                text=f"CHAPTER {chapter_num}",
                font=self.font,
                font_size=self.font_size * 1.5,
                color="white",
                text_align="center",
            ).with_duration(duration).with_position("center")
            
            # Fade in/out
            intro = CompositeVideoClip([bg, txt]).with_fadein(0.5).with_fadeout(0.5)
            logger.info(f"  Creating Intro Card: Chapter {chapter_num}")
            return intro
        except Exception as e:
            logger.debug(f"Failed to render chapter intro: {e}")
            return None

    def _add_subtitles(self, img_clip, text: str, duration: float):
        """
        Add subtitle overlay. Uses the canvas width (self.canvas_w) for
        text-box sizing so captions span the full output frame — not just
        the foreground panel width, which is always smaller on a composited
        canvas. Includes a semi-transparent background strip for legibility
        on any image content.
        """
        from moviepy import TextClip, CompositeVideoClip, ColorClip

        words = text.split()
        if not words:
            return img_clip

        # Scale font proportionally to canvas height so it reads the same
        # on portrait (3:4) and landscape (16:9) output without needing a
        # separate font_size config per aspect ratio.
        font_size = int(self.font_size * (self.canvas_h / 480))
        font_size = max(24, min(font_size, 72))

        box_w = int(self.canvas_w * 0.88)
        chars_per_line = max(10, int(box_w / (font_size * 0.55)))
        max_chars = chars_per_line * 2  # target: ~2 lines per group

        groups: List[str] = []
        group_words: List[List[str]] = []
        current: List[str] = []
        current_len = 0
        for word in words:
            added_len = len(word) + (1 if current else 0)
            if current and current_len + added_len > max_chars:
                groups.append(" ".join(current))
                group_words.append(current)
                current, current_len = [], 0
                added_len = len(word)
            current.append(word)
            current_len += added_len
        if current:
            groups.append(" ".join(current))
            group_words.append(current)

        if not groups:
            return img_clip

        total_words = sum(len(gw) for gw in group_words) or 1
        subtitle_clips = []
        t_cursor = 0.0

        for idx, (g_text, gw) in enumerate(zip(groups, group_words)):
            is_last = idx == len(groups) - 1
            g_dur = max(0.1, duration - t_cursor) if is_last else max(0.1, duration * (len(gw) / total_words))

            txt = TextClip(
                text=g_text,
                font=self.font,
                font_size=font_size,
                color="white",
                stroke_color="black",
                stroke_width=1,
                method="caption",
                size=(box_w, None),
                text_align="center",
            ).with_duration(g_dur).with_start(t_cursor)

            bg_h = txt.h + 24
            margin_bottom = int(self.canvas_h * 0.04)  # 4% from bottom edge
            bg_y = self.canvas_h - bg_h - margin_bottom

            bg = (ColorClip(size=(self.canvas_w, bg_h), color=(0, 0, 0))
                  .with_opacity(0.50)
                  .with_duration(g_dur)
                  .with_start(t_cursor)
                  .with_position(("center", bg_y)))

            txt = txt.with_position(("center", bg_y + 12))
            subtitle_clips.extend([bg, txt])
            t_cursor += g_dur

        return CompositeVideoClip([img_clip] + subtitle_clips)


    def _stitch_final(self, clip_paths: List[str]):
        """
        Use FFmpeg to stitch all rendered clips into the final master video.
        BUG FIX: Uses absolute paths in the concat list to prevent 'file not found' errors.
        """
        if len(clip_paths) == 1:
            import shutil
            shutil.copy2(clip_paths[0], self.final_video_path)
            logger.info(f"Single clip → final: {self.final_video_path}")
            return

        list_path = os.path.join(self.videos_dir, "concat_list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for cp in clip_paths:
                # BUG FIX: Use absolute path to avoid FFmpeg cwd issues
                abs_path = os.path.abspath(cp).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", list_path, "-c", "copy", self.final_video_path],
                check=True,
                capture_output=True,
            )
            os.remove(list_path)
            logger.info(f"✓ Final video: {self.final_video_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg stitch failed: {e.stderr.decode()[:500]}")
        except Exception as e:
            logger.error(f"Stitch error: {e}")
