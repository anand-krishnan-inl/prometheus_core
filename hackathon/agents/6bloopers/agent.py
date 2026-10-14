"""
Bloopers Agent Agent.

Run when you need a laugh or you can't help the user in any other way.

This agent picks a random image from ``/app/images/``, uploads it as an
artifact, and returns the uploaded artifact reference so the user can view it.
"""

import logging
import mimetypes
import random
from pathlib import Path
from typing import Any

from prometheus_api.agent import EventQueue, PrometheusAgent, RequestContext
from prometheus_api.storage_client import ConnectionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_DIR = Path("/app/images")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

class BloopersAgentAgent(PrometheusAgent):
    """Bloopers Agent Agent."""

    def _list_images(self) -> list[Path]:
        """Return all supported image files under /app/images."""
        if not IMAGE_DIR.exists():
            raise FileNotFoundError(f"Image directory does not exist: {IMAGE_DIR}")
        if not IMAGE_DIR.is_dir():
            raise NotADirectoryError(f"Image path is not a directory: {IMAGE_DIR}")

        images = [
            path
            for path in IMAGE_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return images

    async def _execute(
        self,
        params: dict[str, Any],
        context: RequestContext,
        event_queue: EventQueue,
    ) -> dict[str, Any]:
        """Pick, upload, and return a random image from /app/images/."""
        await self.send_status_update(
            context=context,
            event_queue=event_queue,
            message="Picking a random blooper image…",
        )

        try:
            images = self._list_images()
        except Exception as exc:
            logger.exception("Failed to read image directory")
            return {"status": "error", "error": str(exc)}

        if not images:
            return {
                "status": "error",
                "error": f"No image files found in {IMAGE_DIR}. "
                f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}",
            }

        image_path = random.choice(images)
        mime_type, _ = mimetypes.guess_type(str(image_path))
        mime_type = mime_type or "application/octet-stream"

        await self.send_status_update(
            context=context,
            event_queue=event_queue,
            message=f"Uploading {image_path.name}…",
        )

        try:
            ref = await self.send_deeplynx_artifact(
                context=context,
                event_queue=event_queue,
                file_path=image_path,
                name=image_path.name,
                description="Random blooper image",
                storage_config=ConnectionConfig.from_env(),
                mime_type=mime_type,
            )
        except Exception as exc:
            logger.exception("Failed to upload image artifact")
            return {
                "status": "error",
                "error": f"Picked {image_path.name}, but failed to upload it: {exc}",
            }

        return {
            "status": "success",
            "artifact_id": ref.artifact_id,
            "filename": image_path.name,
            "message": "Here is a random image from /app/images/.",
        }

if __name__ == "__main__":
    agent = BloopersAgentAgent(agent_name="Bloopers Agent Agent")
    agent.run()
