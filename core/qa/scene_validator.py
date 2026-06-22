import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class SceneModel(BaseModel):
    scene_id: str
    location: str
    characters: List[str]
    emotion: str
    action: str
    camera_angle: str
    lighting: str
    visual_prompt_tags: str
    narration_text: str
    complexity: int = Field(default=5, ge=0, le=10)
    
    # Optional fields for pipeline state
    chapter: Optional[int] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    reference_images: List[str] = Field(default_factory=list)
    prompt_cache_key: Optional[str] = None
    generation_params: Dict = Field(default_factory=dict)

class SceneValidator:
    """
    Validates scene structure and ensures all required fields are present and valid.
    Ports logic from manhwa-video-factory Stage 07b.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}

    def validate_scenes(self, scenes: List[Dict]) -> List[Dict]:
        valid_scenes = []
        for i, sc in enumerate(scenes):
            try:
                # Ensure scene_id if missing (e.g. from LLM failure)
                if not sc.get("scene_id"):
                    sc["scene_id"] = f"SC{i+1:03d}"
                
                # Validate using Pydantic
                SceneModel(**sc)
                valid_scenes.append(sc)
            except Exception as e:
                logger.warning(f"Scene {sc.get('scene_id', i)} failed validation: {e}")
                # Rejection: previously this still appended the scene, which
                # defeats the purpose of validation. Now we drop it so the
                # orchestrator's coverage check or retry logic can catch it.
                pass 
        
        return valid_scenes
