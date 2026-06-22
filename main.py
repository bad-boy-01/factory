"""
Novel Video Factory v4 — Main Entry Point

Usage:
  python main.py <project_name> [--stage all|translate|memory|char_sheets|visual|generation|audio|video|export]
  python main.py novel --input projects/novel/input/chapter1.txt
  python main.py novel --stage generation           # resume from image generation
  python main.py novel --stage video                # just re-render video
"""
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("NVF")


def main():
    parser = argparse.ArgumentParser(
        description="Novel Video Factory v4 — Korean Manhwa Video from Novel Scripts"
    )
    parser.add_argument("project", help="Project name (must match folder in projects/)")
    parser.add_argument(
        "--stage", default="all",
        choices=["all", "translate", "memory", "char_sheets",
                 "visual", "generation", "audio", "video", "export"],
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--input", default=None,
                        help="Path to novel script (.txt) to import")
    args = parser.parse_args()

    try:
        from core.orchestrator import UnifiedPipeline
        pipeline = UnifiedPipeline(args.project, args.config)

        if args.input:
            pipeline.import_source(args.input)

        stage_map = {
            "translate":   pipeline.stage_translate,
            "memory":      pipeline.stage_memory,
            "char_sheets": pipeline.stage_character_sheets,
            "visual":      pipeline.stage_visual_planning,
            "generation":  pipeline.stage_generation,
            "audio":       pipeline.stage_audio,
            "video":       pipeline.stage_video,
            "export":      pipeline.stage_export,
        }

        if args.stage == "all":
            pipeline.run_all()
        else:
            stage_fn = stage_map.get(args.stage)
            if stage_fn:
                stage_fn()
            else:
                logger.error(f"Unknown stage: {args.stage}")
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
