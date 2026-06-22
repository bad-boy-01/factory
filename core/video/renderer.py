"""
Video Renderer — Novel Video Factory v5
Assembles panels from storyboard.json into the final video.
Supports merge_with_previous by reusing image assets.
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
    try:
        rw, rh = ratio_str.split(":")
        rw, rh = float(rw), float(rh)
    except Exception:
        rw, rh = 3.0, 4.0
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
        self.storyboard_path = os.path.join(self.output_dir, "storyboard.json")
        self.final_video_path = os.path.join(self.videos_dir, "final_video.mp4")

        os.makedirs(self.videos_dir, exist_ok=True)
        logger.info(f"VideoRenderer: canvas={self.canvas_w}x{self.canvas_h} style={self.presentation_style}")

    def render(self):
        if not os.path.exists(self.storyboard_path):
            logger.error("storyboard.json not found — cannot render video")
            return

        with open(self.storyboard_path, "r", encoding="utf-8") as f:
            panels = json.load(f)

        logger.info(f"Rendering chapter with {len(panels)} panels…")
        
        try:
            from moviepy import concatenate_videoclips
        except ImportError:
            logger.error("moviepy not installed.")
            return

        shot_clips = []
        current_img_path = None
        current_chapter = -1

        for panel in panels:
            shot_chapter = panel.get("chapter", 1)
            if shot_chapter != current_chapter:
                intro = self._render_chapter_intro(shot_chapter)
                if intro:
                    shot_clips.append(intro)
                current_chapter = shot_chapter

            pid = panel["id"]
            if panel.get("merge_with_previous", False) and current_img_path:
                img_path = current_img_path
            else:
                img_path = os.path.join(self.images_dir, f"{pid}.png")
                current_img_path = img_path
                
            sc = self._render_shot(panel, img_path)
            if sc is not None:
                shot_clips.append(sc)

        if not shot_clips:
            logger.error("No valid shots to render.")
            return

        try:
            final = concatenate_videoclips(shot_clips, method="compose")
            final.write_videofile(
                self.final_video_path,
                fps=self.fps,
                codec="libx264",
                audio_codec="aac",
                logger=None,
            )
            logger.info(f"✓ Video saved: {self.final_video_path}")
        except Exception as e:
            logger.error(f"Error writing video: {e}")
        finally:
            for sc in shot_clips:
                try:
                    sc.close()
                except Exception:
                    pass
            gc.collect()

        self.generate_srt(panels)

    def generate_srt(self, panels: List[dict]):
        srt_path = os.path.join(self.videos_dir, "subtitles.srt")
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
                for panel in panels:
                    text = panel.get("description", "").strip()
                    if not text:
                        continue

                    pid = panel["id"]
                    aud_path = os.path.join(self.audio_dir, f"{pid}.wav")
                    duration = 3.0
                    if os.path.exists(aud_path):
                        duration = AudioFileClip(aud_path).duration
                    elif panel.get("importance"):
                        imp = panel["importance"]
                        duration = 4.0 if imp >= 9 else (3.0 if imp >= 7 else 1.5)

                    start_str = format_time(current_time)
                    end_str = format_time(current_time + duration)
                    current_time += duration

                    f.write(f"{srt_index}\n{start_str} --> {end_str}\n{text}\n\n")
                    srt_index += 1
            logger.info("✓ SRT generated")
        except Exception as e:
            logger.error(f"Failed to generate SRT: {e}")

    def _render_shot(self, panel: dict, img_path: str):
        try:
            from moviepy import ImageClip, AudioFileClip, CompositeVideoClip
        except ImportError:
            return None

        pid = panel["id"]
        if not os.path.exists(img_path):
            logger.warning(f"Missing image: {img_path} — skipping")
            return None

        audio_clip = None
        aud_path = os.path.join(self.audio_dir, f"{pid}.wav")
        if os.path.exists(aud_path):
            audio_clip = AudioFileClip(aud_path)
            duration = max(audio_clip.duration, 1.0)
        else:
            imp = panel.get("importance", 5)
            duration = 4.0 if imp >= 9 else (3.0 if imp >= 7 else 1.5)

        try:
            from PIL import Image
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception as e:
            logger.error(f"Could not read image size for {pid}: {e}")
            return None

        backdrop_arr = self._build_backdrop(img_path, img_w, img_h)

        beat_type = panel.get("beat_type", "action")
        shot_type = panel.get("shot_type", "medium_shot")
        
        base_speed = self.ken_burns_speed
        speed_mult = 1.0
        if beat_type in ["action", "combat"]: speed_mult = 2.5
        elif beat_type in ["emotion", "reaction"]: speed_mult = 0.8
        elif shot_type == "close_up": speed_mult = 0.5
        elif shot_type in ["wide_shot", "establishing_shot"]: speed_mult = 1.5
        speed = base_speed * speed_mult

        if beat_type == "environment": motion_type = random.choice(["pan_right", "pan_left"])
        elif beat_type == "reveal": motion_type = "zoom_out"
        elif beat_type == "object_focus": motion_type = "zoom_in"
        else: motion_type = random.choice(["zoom_in", "zoom_out", "pan_right", "pan_left"])

        base_scale = min(self.canvas_w / img_w, self.canvas_h / img_h)
        pan_margin = 0.15

        if motion_type in ("pan_left", "pan_right"):
            const_scale = base_scale * (1 + pan_margin)
            clip_w = img_w * const_scale
            clip_h = img_h * const_scale
            max_offset = max(0.0, (clip_w - self.canvas_w) / 2)
            def scale_func(t, _s=const_scale): return _s
            def pos_func(t, _cw=clip_w, _ch=clip_h, _mo=max_offset, _dir=motion_type):
                progress = (t / duration) if duration > 0 else 0.0
                offset = _mo * (1 - 2 * progress) if _dir == "pan_right" else -_mo * (1 - 2 * progress)
                return ((self.canvas_w - _cw) / 2 + offset, (self.canvas_h - _ch) / 2)
        else:
            if motion_type == "zoom_in":
                def scale_func(t, _b=base_scale, _sp=speed): return _b * (1 + (_sp * t / duration if duration > 0 else 0))
            else:
                def scale_func(t, _b=base_scale, _sp=speed): return _b * (1 + _sp - (_sp * t / duration if duration > 0 else 0))
            def pos_func(t, _b=base_scale, _sf=scale_func):
                s = _sf(t)
                cw, ch = img_w * s, img_h * s
                return ((self.canvas_w - cw) / 2, (self.canvas_h - ch) / 2)

        foreground = (ImageClip(img_path)
                      .with_duration(duration)
                      .resized(scale_func)
                      .with_position(pos_func))

        backdrop = ImageClip(backdrop_arr).with_duration(duration)
        composite = CompositeVideoClip([backdrop, foreground], size=(self.canvas_w, self.canvas_h))

        subtitle_text = panel.get("description", "").strip()
        if subtitle_text:
            try:
                composite = self._add_subtitles(composite, subtitle_text, duration)
            except Exception:
                pass

        if audio_clip:
            composite = composite.with_audio(audio_clip)
        return composite

    def _build_backdrop(self, img_path: str, img_w: int, img_h: int):
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
                draw.rectangle([x0 - t, y0 - t, x0 + fit_w + t, y0 + fit_h + t], fill=(20, 20, 20))
            return np.array(canvas)

        cover_scale = max(self.canvas_w / img_w, self.canvas_h / img_h)
        cov_w, cov_h = max(1, int(img_w * cover_scale)), max(1, int(img_h * cover_scale))
        bg = src.resize((cov_w, cov_h), Image.LANCZOS)
        left = (cov_w - self.canvas_w) // 2
        top = (cov_h - self.canvas_h) // 2
        bg = bg.crop((left, top, left + self.canvas_w, top + self.canvas_h))
        bg = bg.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        if self.blur_brightness != 1.0:
            bg = ImageEnhance.Brightness(bg).enhance(self.blur_brightness)
        return np.array(bg)

    def _render_chapter_intro(self, chapter_num: int):
        try:
            from moviepy import TextClip, CompositeVideoClip, ColorClip
            duration = 3.0
            bg = ColorClip(size=(self.canvas_w, self.canvas_h), color=(0, 0, 0)).with_duration(duration)
            txt = TextClip(
                text=f"CHAPTER {chapter_num}",
                font=self.font,
                font_size=self.font_size * 1.5,
                color="white",
                text_align="center",
            ).with_duration(duration).with_position("center")
            intro = CompositeVideoClip([bg, txt]).with_fadein(0.5).with_fadeout(0.5)
            return intro
        except Exception:
            return None

    def _add_subtitles(self, img_clip, text: str, duration: float):
        from moviepy import TextClip, CompositeVideoClip, ColorClip

        words = text.split()
        if not words: return img_clip
        font_size = max(24, min(int(self.font_size * (self.canvas_h / 480)), 72))
        box_w = int(self.canvas_w * 0.88)
        chars_per_line = max(10, int(box_w / (font_size * 0.55)))
        max_chars = chars_per_line * 2

        groups, group_words, current = [], [], []
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

        if not groups: return img_clip
        total_words = sum(len(gw) for gw in group_words) or 1
        subtitle_clips = []
        t_cursor = 0.0

        for idx, (g_text, gw) in enumerate(zip(groups, group_words)):
            is_last = idx == len(groups) - 1
            g_dur = max(0.1, duration - t_cursor) if is_last else max(0.1, duration * (len(gw) / total_words))
            txt = TextClip(
                text=g_text, font=self.font, font_size=font_size,
                color="white", stroke_color="black", stroke_width=1,
                method="caption", size=(box_w, None), text_align="center",
            ).with_duration(g_dur).with_start(t_cursor)

            bg_h = txt.h + 24
            bg_y = self.canvas_h - bg_h - int(self.canvas_h * 0.04)
            bg = (ColorClip(size=(self.canvas_w, bg_h), color=(0, 0, 0))
                  .with_opacity(0.50).with_duration(g_dur).with_start(t_cursor)
                  .with_position(("center", bg_y)))

            txt = txt.with_position(("center", bg_y + 12))
            subtitle_clips.extend([bg, txt])
            t_cursor += g_dur

        return CompositeVideoClip([img_clip] + subtitle_clips)
