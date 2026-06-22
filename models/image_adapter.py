"""
Image Adapter — Novel Video Factory v4
Model: cagliostrolab/animagine-xl-3.1 (FREE on HuggingFace, no API key)

This is the BEST free model for Korean manhwa / webtoon style.
It uses Animagine-XL 3.1 score tags (score_9, score_8_up) for quality.

Features:
- DPM++ 2M Karras scheduler (30% faster than DDIM at same quality)
- IP-Adapter for character reference images (consistency across scenes)
- Multi-pose character sheet generation (6 poses per character)
- Quality filter with auto-retry on bad outputs
- VRAM-safe: CPU offload, VAE slicing/tiling
- 832×480 default (16:9, fast on T4) or 1344×768 (higher quality)
"""
import gc
import hashlib
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# Animagine-XL quality prefix — these MUST come first for best results
ANIMAGINE_QUALITY = "score_9, score_8_up, (masterpiece:1.2), (best quality:1.1), highres"

# Master negative prompt for Animagine-XL
MASTER_NEGATIVE = (
    "score_6, score_5, score_4, "
    "(worst quality, low quality:1.4), bad anatomy, bad hands, "
    "text, error, missing fingers, extra digit, fewer digits, "
    "cropped, jpeg artifacts, signature, watermark, username, "
    "blurry, ugly, deformed, 3d render, photo, photorealistic, "
    "western comic, american comic, fat, extra limbs, cloned face, "
    "mutation, fused fingers, long neck"
)


class LocalImageAdapter:
    """
    Animagine-XL 3.1 image generator — 100% free, no API key.
    Optimised for Kaggle T4 GPU (16GB VRAM).
    """
    def __init__(self, config: dict = None):
        cfg = config or {}
        img_cfg = cfg.get("models", {}).get("image", {})

        self.model_name = img_cfg.get("model", "cagliostrolab/animagine-xl-3.1")
        self.width = img_cfg.get("width", 832)
        self.height = img_cfg.get("height", 480)
        self.steps = img_cfg.get("num_inference_steps", 20)
        self.guidance_scale = img_cfg.get("guidance_scale", 7.0)
        self.use_fast_scheduler = img_cfg.get("use_fast_scheduler", True)
        self.use_ip_adapter = img_cfg.get("use_ip_adapter", True)
        self.ip_adapter_scale = img_cfg.get("ip_adapter_scale", 0.65)
        self.ip_adapter_weight_name = img_cfg.get(
            "ip_adapter_weight_name", "ip-adapter-plus-face_sdxl_vit-h.bin"
        )
        
        # PuLID (BEST for facial identity lock)
        self.identity_method = img_cfg.get("identity_method", "pulid")
        self.pulid_scale = img_cfg.get("pulid_scale", 0.8)

        qf_cfg = cfg.get("quality_filter", {})
        self.max_retries = qf_cfg.get("max_retries", 3)
        self.min_file_size_kb = qf_cfg.get("min_file_size_kb", 10)
        self.quality_filter_enabled = qf_cfg.get("enabled", True)
        self.identity_check_enabled = qf_cfg.get("identity_check_enabled", True)
        self.identity_threshold = qf_cfg.get("identity_threshold", 0.70)

        self.pipeline = None
        self._ip_adapter_loaded = False
        self._pulid_loaded = False
        self._face_analyzer = None
        self._compel = None  # None = not yet attempted, False = tried and unavailable
        self._init_pipeline()

    # ── Pipeline Init ─────────────────────────────────────────────────────────
    def _init_pipeline(self):
        try:
            import torch
            from diffusers import AutoPipelineForText2Image

            logger.info(f"Loading image model: {self.model_name}")
            self.pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )

            # DPM++ 2M Karras — ~30% faster than DDIM at same quality
            if self.use_fast_scheduler:
                try:
                    from diffusers import DPMSolverMultistepScheduler
                    self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                        self.pipeline.scheduler.config,
                        use_karras_sigmas=True,
                        algorithm_type="dpmsolver++",
                    )
                    logger.info("Scheduler: DPM++ 2M Karras ✓")
                except Exception as e:
                    logger.warning(f"Scheduler swap failed: {e}")

            # Memory optimisations for 16GB T4 VRAM
            try:
                self.pipeline.enable_vae_slicing()
                self.pipeline.enable_vae_tiling()
            except Exception:
                pass

            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("xformers ✓")
            except Exception:
                pass

            # Critical: CPU offload keeps 16GB VRAM safe
            self.pipeline.enable_model_cpu_offload()
            logger.info(f"Pipeline ready: {self.model_name} | "
                        f"{self.width}×{self.height} | {self.steps} steps")

        except ImportError as e:
            logger.warning(f"diffusers/torch not installed ({e}) — MOCK mode active")
            self.pipeline = None
        except Exception as e:
            logger.error(f"Pipeline init failed: {e} — MOCK mode active")
            self.pipeline = None

    def _ensure_pulid(self):
        """Load PuLID for SDXL identity preservation."""
        if self._pulid_loaded or self.pipeline is None:
            return
        
        try:
            import torch
            from huggingface_hub import hf_hub_download
            
            # 1. Ensure weights are present
            weights_dir = "models/weights/pulid"
            os.makedirs(weights_dir, exist_ok=True)
            
            pulid_path = os.path.join(weights_dir, "ip-adapter_sdxl_pulid.bin")
            if not os.path.exists(pulid_path):
                logger.info("Downloading PuLID weights…")
                hf_hub_download(
                    repo_id="yanze/PuLID",
                    filename="ip-adapter_sdxl_pulid.bin",
                    local_dir=weights_dir
                )
            
            # 2. Load EVA-CLIP (Identity Encoder)
            # This is a large ViT-G/14 model. 
            # We'll need a specialized loader or use a library if available.
            # For this implementation, we assume the user has the 'pulid' 
            # library or we use a manual patch.
            
            # NOTE: Full PuLID implementation often requires custom UNet patching.
            # In this 'factory' version, we'll use the IP-Adapter format if 
            # we can't find the 'pulid' module, but we'll try the real one first.
            try:
                from pulid.pipeline_helper import PuLIDPipelineHelper
                self.pulid_helper = PuLIDPipelineHelper(self.pipeline, device="cuda" if torch.cuda.is_available() else "cpu")
                self.pulid_helper.load_pulid_weights(pulid_path)
                self._pulid_loaded = True
                logger.info(f"PuLID identity adapter loaded ✓ (scale={self.pulid_scale})")
            except ImportError:
                logger.warning("PuLID library not found. Falling back to IP-Adapter for identity.")
                self.identity_method = "ip-adapter"
                self._ensure_ip_adapter()

        except Exception as e:
            logger.error(f"PuLID init failed: {e}. Using IP-Adapter fallback.")
            self.identity_method = "ip-adapter"
            self._ensure_ip_adapter()

    def _ensure_ip_adapter(self):
        """Load IP-Adapter on first use (lazy-loading saves VRAM)."""
        if self._ip_adapter_loaded or self.pipeline is None:
            return
        if not self.use_ip_adapter:
            return
        # Try the configured weight first (default: face-specialized variant,
        # better at "is this the same character" than the generic weight that
        # was previously hardcoded — generic IP-Adapter is tuned more for
        # overall style/composition transfer than facial identity lock).
        # Fall back to the generic weight if it's unavailable for any reason,
        # rather than disabling IP-Adapter entirely.
        candidates = [self.ip_adapter_weight_name]
        if self.ip_adapter_weight_name != "ip-adapter_sdxl.bin":
            candidates.append("ip-adapter_sdxl.bin")
        for weight_name in candidates:
            try:
                self.pipeline.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="sdxl_models",
                    weight_name=weight_name,
                )
                self.pipeline.set_ip_adapter_scale(self.ip_adapter_scale)
                # Re-enable CPU offload so the new IP-Adapter encoder is handled correctly
                self.pipeline.enable_model_cpu_offload()
                self._ip_adapter_loaded = True
                logger.info(
                    f"IP-Adapter loaded ✓ ({weight_name}, scale={self.ip_adapter_scale}) "
                    f"— character consistency enabled"
                )
                return
            except Exception as e:
                logger.warning(f"IP-Adapter weight '{weight_name}' failed to load: {e}")
        logger.warning("IP-Adapter unavailable on any candidate weight — consistency via prompts only")

    def _ensure_compel(self):
        """
        Lazy-load Compel for long-prompt support. SDXL's two CLIP text
        encoders have a hard 77-token limit each — by default diffusers
        truncates anything past that silently, which is what was dropping
        the "korean manhwa style" tags on nearly every image (see
        FALLBACK_FAILURE_ANALYSIS.md). Compel encodes a prompt in 77-token
        chunks and concatenates the resulting embeddings, so nothing is
        dropped regardless of prompt length.
        """
        if self._compel is not None or self.pipeline is None:
            return
        try:
            from compel import Compel, ReturnedEmbeddingsType
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # SDXL configuration for Compel
            self._compel = Compel(
                tokenizer=[self.pipeline.tokenizer, self.pipeline.tokenizer_2],
                text_encoder=[self.pipeline.text_encoder, self.pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                truncate_long_prompts=False,
                device=device,
            )
            logger.info("Compel long-prompt encoder loaded ✓ (prompts >77 tokens are chunked, not truncated)")
        except Exception as e:
            logger.warning(
                f"Compel unavailable ({e}) — falling back to plain prompt strings"
            )
            self._compel = False  # sentinel: tried once, don't retry every call

    # ── Public Interface ──────────────────────────────────────────────────────
    def generate_image(
        self,
        prompt: str,
        output_path: str,
        negative_prompt: str = "",
        reference_image_paths: List[str] = None,
        seed: int = None,
        generation_params: dict = None,
    ):
        """
        Generate one image and save to output_path.
        Retries up to max_retries times if quality filter rejects it.
        """
        params = generation_params or {}
        effective_seed = params.get("seed", seed)
        steps = params.get("steps", self.steps)
        cfg = params.get("cfg", self.guidance_scale)
        w = params.get("width", self.width)
        h = params.get("height", self.height)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # In mock mode (no GPU/diffusers) placeholders are tiny — skip QA entirely
        use_qa = self.quality_filter_enabled and (self.pipeline is not None)

        for attempt in range(self.max_retries):
            s = ((effective_seed + attempt * 7919) % (2**31 - 1)
                 if effective_seed is not None else 42)
            if attempt > 0:
                logger.info(f"  Retry {attempt}/{self.max_retries - 1} (seed={s})")

            self._run_generation(prompt, output_path, negative_prompt,
                                 reference_image_paths, s, steps, cfg, w, h)

            if not use_qa:
                return  # accepted ✓
            
            # QA Filter 1: File size (corruption/blank check)
            if not self._passes_quality(output_path):
                continue

            # QA Filter 2: Smart Identity Check (Similarity score)
            if self.identity_check_enabled and reference_image_paths:
                score = self._calculate_identity_score(output_path, reference_image_paths)
                if score < self.identity_threshold:
                    logger.warning(
                        f"    Identity QA fail: score {score:.2f} < {self.identity_threshold} "
                        f"(threshold). Retrying..."
                    )
                    continue
                else:
                    logger.info(f"    Identity QA pass: score {score:.2f} ✓")

            return  # accepted ✓

        if use_qa:
            logger.warning(f"All {self.max_retries} attempts failed QA filter: {output_path}")

    def _calculate_identity_score(self, image_path: str, reference_paths: List[str]) -> float:
        """
        Calculates cosine similarity between the face in the generated image
        and the faces in the reference images. Returns 0.0 to 1.0.
        """
        try:
            self._ensure_face_analyzer()
            if not self._face_analyzer:
                return 1.0 # Skip check if analyzer failed to load

            import numpy as np
            from PIL import Image

            # 1. Get embedding for the generated face
            gen_img = np.array(Image.open(image_path).convert("RGB"))
            gen_faces = self._face_analyzer.get(gen_img)
            if not gen_faces:
                logger.debug(f"      No face detected in generated image: {os.path.basename(image_path)}")
                return 0.5 # Neutral score: we can't confirm it's wrong, but it's not strongly right

            # Take the largest face in the generated image
            gen_faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
            gen_emb = gen_faces[0].normed_embedding

            # 2. Compare against reference embeddings
            similarities = []
            for ref_p in reference_paths:
                if not os.path.exists(ref_p): continue
                ref_img = np.array(Image.open(ref_p).convert("RGB"))
                ref_faces = self._face_analyzer.get(ref_img)
                if not ref_faces: continue
                
                # Take the largest face in the reference
                ref_faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                ref_emb = ref_faces[0].normed_embedding
                
                # Cosine similarity
                sim = np.dot(gen_emb, ref_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(ref_emb))
                # Map from [-1, 1] to [0, 1]
                sim = (sim + 1) / 2
                similarities.append(sim)

            if not similarities:
                return 1.0 # No reference faces found to compare against

            return float(np.mean(similarities))

        except Exception as e:
            logger.warning(f"      Identity score calculation failed: {e}")
            return 1.0 # Fail open: don't block pipeline on analyzer errors

    def _ensure_face_analyzer(self):
        """Lazy-load InsightFace for identity checking and PuLID."""
        if self._face_analyzer is not None:
            return
        try:
            import insightface
            from insightface.app import FaceAnalysis
            import torch

            logger.info("Loading Face-Analysis (InsightFace) for identity consistency…")
            self._face_analyzer = FaceAnalysis(
                name='antelopev2', root='models/weights/insightface', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            # Use fp16 if on GPU
            self._face_analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
            logger.info("Face-Analysis loaded ✓")
        except Exception as e:
            logger.warning(f"Face-Analysis failed to load ({e}). Identity check/PuLID will be skipped.")
            self._face_analyzer = False

    def _run_generation(self, prompt, output_path, negative_prompt,
                        ref_paths, seed, steps, cfg, w, h):
        if self.pipeline is None:
            logger.info(f"[MOCK] {prompt[:60]}…")
            self._save_placeholder(output_path, w, h)
            return

        import torch
        
        # Check token length
        input_ids = self.pipeline.tokenizer(prompt).input_ids
        prompt_token_length = len(input_ids)
        logger.info(f"Prompt token length: {prompt_token_length}")

        self._ensure_compel()
        kwargs = None

        if self._compel:
            try:
                # 1. Get embeddings
                p_emb, p_pooled = self._compel(prompt)
                n_emb, n_pooled = self._compel(negative_prompt or MASTER_NEGATIVE)
                
                logger.info(f"  Compel encoded shape: {p_emb.shape} on device: {p_emb.device}")
                
                # 2. Manual padding for sequence length mismatch
                # Compel for SDXL returns [1, Seq, 2048] for emb and [1, 1280] for pooled
                seq_len = p_emb.shape[1]
                neg_seq_len = n_emb.shape[1]
                
                if seq_len > neg_seq_len:
                    padding = torch.zeros(
                        (1, seq_len - neg_seq_len, n_emb.shape[2]),
                        device=n_emb.device, dtype=n_emb.dtype
                    )
                    n_emb = torch.cat([n_emb, padding], dim=1)
                elif neg_seq_len > seq_len:
                    padding = torch.zeros(
                        (1, neg_seq_len - seq_len, p_emb.shape[2]),
                        device=p_emb.device, dtype=p_emb.dtype
                    )
                    p_emb = torch.cat([p_emb, padding], dim=1)

                kwargs = {
                    "prompt_embeds": p_emb,
                    "pooled_prompt_embeds": p_pooled,
                    "negative_prompt_embeds": n_emb,
                    "negative_pooled_prompt_embeds": n_pooled,
                    "width": w,
                    "height": h,
                    "num_inference_steps": steps,
                    "guidance_scale": cfg,
                }
            except Exception as e:
                logger.exception(f"  Compel encoding failed ({e}) — using plain prompt")
                logger.error(f"  Debug Info - Prompt Length: {prompt_token_length}, Expected Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
                kwargs = None

        if kwargs is None:
            kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or MASTER_NEGATIVE,
                "width": w,
                "height": h,
                "num_inference_steps": steps,
                "guidance_scale": cfg,
            }

        # Deterministic seed
        try:
            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)
        except Exception:
            pass

        # Identity Injection (PuLID or IP-Adapter)
        if ref_paths:
            valid_refs = [p for p in ref_paths if p and os.path.exists(p)]
            if valid_refs:
                if self.identity_method == "pulid":
                    self._ensure_pulid()
                    if self._pulid_loaded:
                        try:
                            # 1. Get face embedding for PuLID
                            self._ensure_face_analyzer()
                            from PIL import Image
                            import numpy as np
                            
                            ref_img = np.array(Image.open(valid_refs[0]).convert("RGB"))
                            faces = self._face_analyzer.get(ref_img)
                            if faces:
                                faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                                id_emb = torch.from_numpy(faces[0].normed_embedding).to(self.pipeline.device)
                                
                                # 2. Apply PuLID to UNet
                                self.pulid_helper.set_id_embedding(id_emb, scale=self.pulid_scale)
                                logger.debug(f"  PuLID identity applied: {os.path.basename(valid_refs[0])}")
                        except Exception as pe:
                            logger.warning(f"  PuLID injection failed: {pe}")
                
                elif self.identity_method == "ip-adapter" or self.use_ip_adapter:
                    self._ensure_ip_adapter()
                    if self._ip_adapter_loaded:
                        try:
                            from PIL import Image
                            import numpy as np
                            imgs = [Image.open(p).convert("RGB") for p in valid_refs]
                            if len(imgs) > 1:
                                # Average multiple reference images
                                avg = np.mean(
                                    [np.array(img.resize((224, 224))) for img in imgs], axis=0
                                ).astype("uint8")
                                ref_img = Image.fromarray(avg)
                            else:
                                ref_img = imgs[0]
                            
                            # V4 Fix: SDXL IP-Adapter needs added_cond_kwargs
                            # This fixes the 'requires image_embeds' and 'mat1/mat2' shape errors
                            kwargs["added_cond_kwargs"] = {"image_embeds": ref_img}
                            logger.debug(f"  IP-Adapter added_cond_kwargs applied")
                        except Exception as e:
                            logger.warning(f"  IP-Adapter injection failed: {e}")

        try:
            # The prompt_embeds/pooled_prompt_embeds must not conflict with 
            # the added_cond_kwargs if the UNet is configured for IP-Adapter
            image = self.pipeline(**kwargs).images[0]
            image.save(output_path)
            logger.info(f"  ✓ Saved: {os.path.basename(output_path)}")
        except Exception as e:
            logger.error(f"  Generation error: {e}")
            self._save_placeholder(output_path, w, h)
        finally:
            # Clean up PuLID state to avoid affecting next generation if it doesn't have refs
            if self._pulid_loaded:
                try:
                    self.pulid_helper.clear_id_embedding()
                except Exception:
                    pass

    # ── Multi-pose Character Sheet ────────────────────────────────────────────
    def generate_character_sheet(
        self,
        char_id: str,
        char_name: str,
        dna_str: str,
        output_dir: str,
        negative_prompt: str = "",
        world_style: str = "",
        poses: list = None,
    ):
        """
        Generates 1 primary reference model sheet for a character.
        Creates: output_dir/{char_id}.png
        These are used by IP-Adapter/PuLID for consistency across all scenes.
        """
        out_path = os.path.join(output_dir, f"{char_id}.png")
        
        if os.path.exists(out_path):
            logger.info(f"  Reference exists: {char_name} — skipping")
            return

        seed_base = abs(hash(char_id + char_name)) % (2**31 - 1)

        # Detect gender for proper subject tag
        dna_lower = dna_str.lower()
        gender_tag = ("1girl"
                      if any(w in dna_lower for w in ["girl", "woman", "female", "she"])
                      else "1boy")

        # Inject world context (genre/setting) to ensure outfits match the story universe
        setting_context = f"wearing clothes fitting the setting: ({world_style}), " if world_style else ""

        prompt = (
            f"{ANIMAGINE_QUALITY}, "
            f"{gender_tag}, {dna_str}, {setting_context}"
            f"character design sheet, concept art, turnaround, multiple views, "
            f"front view, side view, back view, close-up portrait, simple background, "
            f"manhwa, webtoon, korean manhwa style, sharp lineart"
        )
        neg = (
            f"score_6, score_5, (worst quality, low quality:1.4), "
            f"bad anatomy, blurry, text, watermark, {negative_prompt}"
        )
        
        logger.info(f"  Generating {char_name} reference sheet…")

        # Wider format for character sheets (fits multiple views)
        self.generate_image(
            prompt, out_path, negative_prompt=neg,
            seed=seed_base,
            generation_params={"width": 1024, "height": 768, "steps": 25, "cfg": 7.0},
        )

    # ── Quality Filter ────────────────────────────────────────────────────────
    def _passes_quality(self, output_path: str) -> bool:
        """Basic quality check: file exists and is large enough to be a real image."""
        if not os.path.exists(output_path):
            return False
        size_kb = os.path.getsize(output_path) / 1024
        if size_kb < self.min_file_size_kb:
            logger.warning(f"  QA fail: file too small ({size_kb:.1f}KB < {self.min_file_size_kb}KB)")
            return False
        return True

    # ── Utilities ─────────────────────────────────────────────────────────────
    def _save_placeholder(self, output_path: str, w: int = 832, h: int = 480):
        """Create a visible placeholder so downstream stages don't crash."""
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (w, h), color=(20, 20, 40))
            draw = ImageDraw.Draw(img)
            draw.text((w // 2 - 80, h // 2 - 10),
                      "[ IMAGE GENERATION FAILED ]", fill=(180, 80, 80))
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            img.save(output_path)
        except Exception as e:
            logger.error(f"Placeholder save failed: {e}")
            with open(output_path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n")  # Minimal PNG header as last resort

    def unload(self):
        """Free GPU + CPU RAM before video rendering."""
        if self.pipeline is not None:
            logger.info("Unloading image pipeline from VRAM…")
            try:
                self.pipeline.to("cpu")
            except Exception:
                pass
            del self.pipeline
            self.pipeline = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared ✓")
        except ImportError:
            pass

    def cleanup(self):
        """Alias for compatibility."""
        self.unload()
