"""
title: GPT-Image-1
description: Simple plugin for generating and editing images using OpenAI's GPT-Image-1 model
author: Shayano
author_url: https://github.com/Shayano/
funding_url: https://github.com/open-webui
version: 1.1.0
"""

import base64
import io
import json
import logging
import re
import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Awaitable, Literal
import httpx
from fastapi import Request, UploadFile, BackgroundTasks
from open_webui.routers.files import upload_file, get_file_content_by_id
from open_webui.models.users import UserModel, Users
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from starlette.datastructures import Headers

logger = logging.getLogger(__name__)


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default="", description="Your OpenAI API key")
        API_BASE_URL: str = Field(
            default="https://api.openai.com/v1", description="OpenAI API base URL"
        )
        IMAGE_COUNT: int = Field(
            default=1, description="Number of images to generate (1-10)", ge=1, le=10
        )
        IMAGE_QUALITY: Literal["low", "medium", "high"] = Field(
            default="medium", description="Quality of the generated image"
        )
        IMAGE_SIZE: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = Field(
            default="1024x1024", description="Size of the generated image"
        )
        MODERATION: Literal["auto", "low"] = Field(
            default="auto", description="Moderation strictness"
        )
        REQUEST_TIMEOUT: int = Field(
            default=600, description="Request timeout in seconds"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.emitter: Optional[Callable[[dict], Awaitable[None]]] = None

    async def emit_status(
        self, message: str, done: bool = False, show_in_chat: bool = False
    ):
        """Emit status updates to the client."""
        if self.emitter:
            await self.emitter(
                {"type": "status", "data": {"description": message, "done": done}}
            )
        # Only return a message for the chat if show_in_chat is True
        if show_in_chat:
            if done:
                return f"**✅ {message}**\n\n"
            else:
                return f"**⏳ {message}**\n\n"
        return ""

    async def pipes(self) -> List[dict]:
        return [{"id": "gpt-image-1", "name": "GPT Image 1"}]

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __request__: Request,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> StreamingResponse:
        self.emitter = __event_emitter__
        user = Users.get_user_by_id(__user__["id"])

        async def stream_response():
            try:
                model = body.get("model", "gpt-image-1")
                messages = body.get("messages", [])
                is_stream = body.get("stream", False)
                # Extract prompt and images from messages
                prompt, images = await self._extract_prompt_and_images(messages)
                if not self.valves.OPENAI_API_KEY:
                    yield self._format_data(
                        is_stream=is_stream,
                        content="Error: OpenAI API key not provided. Please set the OPENAI_API_KEY valve.",
                    )
                    return
                # Determine if we're generating or editing
                if images:
                    await self.emit_status("✂️ Preparing image editing...")
                    endpoint = "/images/edits"
                else:
                    await self.emit_status("🖼️ Preparing image generation...")
                    endpoint = "/images/generations"
                # Make the request to OpenAI
                try:
                    async with httpx.AsyncClient(
                        base_url=self.valves.API_BASE_URL,
                        headers={
                            "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}"
                        },
                        timeout=self.valves.REQUEST_TIMEOUT,
                    ) as client:
                        if images:
                            # For image edits
                            form_data = {}
                            files = {}
                            # Add image file
                            image_data = base64.b64decode(images[0]["data"])
                            files["image"] = (
                                f"image.{images[0]['mimeType'].split('/')[-1]}",
                                image_data,
                                images[0]["mimeType"],
                            )
                            # Add other parameters
                            form_data.update(
                                {
                                    "prompt": prompt,
                                    "n": str(min(self.valves.IMAGE_COUNT, 10)),
                                    "size": (
                                        self.valves.IMAGE_SIZE
                                        if self.valves.IMAGE_SIZE != "auto"
                                        else "1024x1024"
                                    ),
                                    "quality": self.valves.IMAGE_QUALITY,
                                    "model": model.split(".", 1)[-1],
                                }
                            )
                            if self.valves.MODERATION != "auto":
                                form_data["moderation"] = self.valves.MODERATION
                            await self.emit_status(
                                "🔄 Editing image... This may take a moment."
                            )
                            try:
                                response = await client.post(
                                    endpoint, data=form_data, files=files
                                )
                                response.raise_for_status()
                            except httpx.HTTPError as e:
                                logger.error(f"OpenAI API request failed: {e}")
                                error_status = await self.emit_status(
                                    f"❌ An error occurred while editing image",
                                    True,
                                    True,
                                )
                                yield self._format_data(
                                    is_stream=is_stream,
                                    content=f"{error_status}Error editing image: {str(e)}",
                                )
                                return
                        else:
                            # For image generations
                            json_data = {
                                "prompt": prompt,
                                "n": min(self.valves.IMAGE_COUNT, 10),
                                "size": (
                                    self.valves.IMAGE_SIZE
                                    if self.valves.IMAGE_SIZE != "auto"
                                    else "1024x1024"
                                ),
                                "quality": self.valves.IMAGE_QUALITY,
                                "model": model.split(".", 1)[-1],
                            }
                            if self.valves.MODERATION != "auto":
                                json_data["moderation"] = self.valves.MODERATION
                            await self.emit_status(
                                "🔄 Generating image... This may take a moment."
                            )
                            try:
                                response = await client.post(endpoint, json=json_data)
                                response.raise_for_status()
                            except httpx.HTTPError as e:
                                logger.error(f"OpenAI API request failed: {e}")
                                error_status = await self.emit_status(
                                    f"❌ An error occurred while generating image",
                                    True,
                                    True,
                                )
                                yield self._format_data(
                                    is_stream=is_stream,
                                    content=f"{error_status}Error generating image: {str(e)}",
                                )
                                return
                        response_data = response.json()
                        # Process images - don't show this status in chat
                        await self.emit_status("✅ Image processing complete!", True)
                        image_markdown = []
                        for i, img_data in enumerate(response_data.get("data", [])):
                            if "b64_json" in img_data:
                                try:
                                    image_url = self._upload_image(
                                        __request__=__request__,
                                        user=user,
                                        image_data=img_data["b64_json"],
                                        mime_type="image/png",
                                    )
                                    image_markdown.append(
                                        f"![image_{i+1}]({image_url})"
                                    )
                                except Exception as e:
                                    logger.error(f"Image upload failed: {e}")
                                    error_status = await self.emit_status(
                                        f"❌ An error occurred while uploading image",
                                        True,
                                        True,
                                    )
                                    yield self._format_data(
                                        is_stream=is_stream,
                                        content=f"{error_status}Error uploading image: {str(e)}",
                                    )
                                    return
                            elif "url" in img_data:
                                image_markdown.append(
                                    f"![image_{i+1}]({img_data['url']})"
                                )
                        # Combine all images into one response
                        content = "\n\n".join(image_markdown)
                        # Send response
                        if is_stream:
                            yield self._format_data(
                                is_stream=True,
                                model=model,
                                content=content,
                            )
                            # Send a final message with usage if available
                            yield self._format_data(
                                is_stream=True,
                                model=model,
                                content=None,
                                usage=response_data.get("usage"),
                            )
                        else:
                            yield self._format_data(
                                is_stream=False,
                                model=model,
                                content=content,
                                usage=response_data.get("usage"),
                            )
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    error_status = await self.emit_status(
                        f"❌ An unexpected error occurred", True, True
                    )
                    yield self._format_data(
                        is_stream=is_stream,
                        content=f"{error_status}Unexpected error: {str(e)}",
                    )
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                error_status = await self.emit_status(
                    f"❌ An error occurred while processing request", True, True
                )
                yield self._format_data(
                    is_stream=body.get("stream", False),
                    content=f"{error_status}Error processing request: {str(e)}",
                )

        return StreamingResponse(stream_response())

    async def _extract_prompt_and_images(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, str]]]:
        """Extract prompt and image data from messages."""
        prompt = ""
        images = []

        # First, get the prompt from the most recent user message
        for message in reversed(messages):
            if message.get("role") != "user":
                continue

            content = message.get("content", "")

            # Extract text prompt
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        prompt += item.get("text", "") + " "
            elif isinstance(content, str):
                prompt = content

            # We have our prompt, break here
            break

        prompt = prompt.strip()

        # Now look for images in ALL messages (both user and assistant), starting from the most recent
        for message in reversed(messages):
            # Check both user messages (uploaded images) and assistant messages (generated images)
            role = message.get("role", "")
            if role not in ["user", "assistant"]:
                continue

            content = message.get("content", "")

            # Debug logging
            logger.debug(
                f"Checking {role} message for images. Content type: {type(content)}"
            )

            # If content is a list (mixed content format - typically in user messages)
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        logger.debug(
                            f"Found image_url item. URL starts with: {url[:50] if url else 'empty'}"
                        )

                        # Handle both data URLs and file URLs
                        if url.startswith("data:"):
                            parts = url.split(";base64,", 1)
                            if len(parts) == 2:
                                mime_type = parts[0].replace("data:", "", 1)
                                data = parts[1]
                                images.append({"mimeType": mime_type, "data": data})
                                logger.debug(
                                    f"Added data URL image with mime type: {mime_type}"
                                )
                        elif "/api/v1/files/" in url or "/files/" in url:
                            # This is a file reference, we need to fetch it
                            # Extract file ID more carefully
                            if "/api/v1/files/" in url:
                                file_id = (
                                    url.split("/api/v1/files/")[-1]
                                    .split("/")[0]
                                    .split("?")[0]
                                )
                            else:
                                file_id = (
                                    url.split("/files/")[-1].split("/")[0].split("?")[0]
                                )

                            logger.debug(f"Found file reference with ID: {file_id}")
                            try:
                                # Try to fetch using the Files model directly
                                from open_webui.models.files import Files

                                file_item = Files.get_file_by_id(file_id)
                                if file_item:
                                    # Read the file data
                                    file_path = file_item.path
                                    if file_path:
                                        with open(file_path, "rb") as f:
                                            file_data = f.read()
                                        data = base64.b64encode(file_data).decode(
                                            "utf-8"
                                        )
                                        mime_type = file_item.meta.get(
                                            "content_type", "image/png"
                                        )
                                        images.append(
                                            {"mimeType": mime_type, "data": data}
                                        )
                                        logger.debug(
                                            f"Successfully fetched and converted file {file_id}"
                                        )
                            except Exception as e:
                                logger.error(f"Failed to fetch file {file_id}: {e}")

            # If content is a string (could be in both user and assistant messages)
            elif isinstance(content, str):
                # Look for inline image markdown format: ![alt](data:mime/type;base64,DATA)
                pattern = r"!\[[^\]]*\]\(data:([^;]+);base64,([^)]+)\)"
                matches = re.findall(pattern, content)
                if matches:
                    logger.debug(
                        f"Found {len(matches)} inline image(s) in {role} string content"
                    )
                for mime_type, data in matches:
                    images.append({"mimeType": mime_type, "data": data})

                # Also check for file references in markdown format: ![alt](/api/v1/files/ID)
                # This is where generated images typically appear in assistant messages
                file_pattern = r"!\[[^\]]*\]\((/api/v1/files/[^)]+|/files/[^)]+)\)"
                file_matches = re.findall(file_pattern, content)
                if file_matches:
                    logger.debug(
                        f"Found {len(file_matches)} file reference(s) in {role} string content"
                    )
                for file_url in file_matches:
                    # Extract file ID more carefully
                    if "/api/v1/files/" in file_url:
                        file_id = (
                            file_url.split("/api/v1/files/")[-1]
                            .split("/")[0]
                            .split("?")[0]
                        )
                    else:
                        file_id = (
                            file_url.split("/files/")[-1].split("/")[0].split("?")[0]
                        )

                    logger.debug(f"Extracted file ID: {file_id} from URL: {file_url}")
                    try:
                        # Try to fetch using the Files model directly
                        from open_webui.models.files import Files

                        file_item = Files.get_file_by_id(file_id)
                        if file_item:
                            # Read the file data
                            file_path = file_item.path
                            if file_path:
                                with open(file_path, "rb") as f:
                                    file_data = f.read()
                                data = base64.b64encode(file_data).decode("utf-8")
                                mime_type = file_item.meta.get(
                                    "content_type", "image/png"
                                )
                                images.append({"mimeType": mime_type, "data": data})
                                logger.debug(
                                    f"Successfully fetched file {file_id} from {role} markdown"
                                )
                    except Exception as e:
                        logger.error(f"Failed to fetch file {file_id}: {e}")

            # If we found images, we can stop searching
            if images:
                logger.debug(
                    f"Found {len(images)} image(s) in {role} message, stopping search"
                )
                break

        # Clean up the prompt if it contains image markdown (only for the current message)
        if images and isinstance(prompt, str):
            pattern = r"!\[[^\]]*\]\(data:[^;]+;base64,[^)]+\)"
            prompt = re.sub(pattern, "", prompt).strip()
            # Also remove file reference patterns
            file_pattern = r"!\[[^\]]*\]\((/api/v1/files/[^)]+|/files/[^)]+)\)"
            prompt = re.sub(file_pattern, "", prompt).strip()

        logger.info(
            f"Extracted prompt: '{prompt[:100]}...', found {len(images)} image(s)"
        )

        return prompt, images

    def _upload_image(
        self, __request__: Request, user: UserModel, image_data: str, mime_type: str
    ) -> str:
        try:
            file_item = upload_file(
                request=__request__,
                background_tasks=BackgroundTasks(),
                file=UploadFile(
                    file=io.BytesIO(base64.b64decode(image_data)),
                    filename=f"generated-image-{uuid.uuid4().hex}.png",
                    headers=Headers({"content-type": mime_type}),
                ),
                process=False,
                user=user,
                metadata={"mime_type": mime_type},
            )
            image_url = __request__.app.url_path_for(
                "get_file_content_by_id", id=file_item.id
            )
            return image_url
        except Exception as e:
            logger.error(f"Image upload failed: {e}")
            raise

    def _format_data(
        self,
        is_stream: bool,
        model: str = "",
        content: Optional[str] = "",
        usage: Optional[dict] = None,
    ) -> str:
        """Format the response data in the expected OpenAI-compatible format."""
        data = {
            "id": f"chat.{uuid.uuid4().hex}",
            "object": "chat.completion.chunk" if is_stream else "chat.completion",
            "created": int(time.time()),
            "model": model,
        }
        if content:
            data["choices"] = [
                {
                    "finish_reason": "stop" if not is_stream else None,
                    "index": 0,
                    "delta" if is_stream else "message": {
                        "role": "assistant",
                        "content": content,
                    },
                }
            ]
        if usage:
            data["usage"] = usage
        if is_stream:
            return f"data: {json.dumps(data)}\n\n"
        else:
            return json.dumps(data)
