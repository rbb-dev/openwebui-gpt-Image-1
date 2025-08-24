"""
title: GPT-Image-1
description: Simple plugin for generating and editing images using OpenAI's GPT-Image-1 model
author: Shayano  +  upload extension
version: 1.2.0
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
from fastapi import BackgroundTasks, Request, UploadFile
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from open_webui.models.users import Users
from open_webui.routers.files import upload_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Pipe:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(default="", description="Your OpenAI API key")
        API_BASE_URL: str = Field(default="https://api.openai.com/v1", description="OpenAI API base URL")
        IMAGE_COUNT: int = Field(default=1, ge=1, le=10, description="Number of images to generate")
        IMAGE_QUALITY: Literal["low", "medium", "high"] = Field(default="medium", description="Quality")
        IMAGE_SIZE: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = Field(
            default="1024x1024", description="Size of the generated image"
        )
        MODERATION: Literal["auto", "low"] = Field(default="auto", description="Moderation strictness")
        REQUEST_TIMEOUT: int = Field(default=600, description="Request timeout in seconds")
        SAVE_TO_FILES: bool = Field(
            default=False, description="Save generated images to the Open-WebUI file-store"
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

        async def stream_response():
            try:
                model = body.get("model", "gpt-image-1")
                messages = body.get("messages", [])
                is_stream = bool(body.get("stream", False))
                prompt, images = self._extract_prompt_and_images(messages)

                if not self.valves.OPENAI_API_KEY:
                    yield self._format_data(
                        is_stream=is_stream,
                        content="Error: OpenAI API key not provided.",
                    )
                    return

                endpoint = "/images/edits" if images else "/images/generations"
                async with httpx.AsyncClient(
                    base_url=self.valves.API_BASE_URL,
                    headers={"Authorization": f"Bearer {self.valves.OPENAI_API_KEY}"},
                    timeout=self.valves.REQUEST_TIMEOUT,
                ) as client:
                    if images:
                        form_data = {
                            "prompt": prompt,
                            "n": min(self.valves.IMAGE_COUNT, 10),
                            "size": (
                                self.valves.IMAGE_SIZE
                                if self.valves.IMAGE_SIZE != "auto"
                                else "1024x1024"
                            ),
                            "model": model.split(".", 1)[-1],
                            "quality": self.valves.IMAGE_QUALITY,
                        }
                        files = {"image": self._prep_file(images[0])}  # first image only
                        resp = await client.post(endpoint, data=form_data, files=files)
                    else:
                        json_data = {
                            "prompt": prompt,
                            "n": min(self.valves.IMAGE_COUNT, 10),
                            "model": model.split(".", 1)[-1],
                            "size": (
                                self.valves.IMAGE_SIZE
                                if self.valves.IMAGE_SIZE != "auto"
                                else "1024x1024"
                            ),
                            "quality": self.valves.IMAGE_QUALITY,
                        }
                        resp = await client.post(endpoint, json=json_data)

                    if resp.status_code != 200:
                        msg = f"OpenAI error {resp.status_code}: {resp.text}"
                        yield self._format_data(is_stream=is_stream, content=msg)
                        return

                    data = resp.json()
                    image_markup = self._build_image_markup(data)

                    if self.valves.SAVE_TO_FILES:
                        image_markup = await self._upload_and_return_links(
                            __request__, __user__, image_markup
                        )

                    content = "\n\n".join(image_markup)
                    if is_stream:
                        yield self._format_data(
                            is_stream=True, model=model, content=content
                        )
                        yield self._format_data(
                            is_stream=True, model=model, content=None, usage=data.get("usage")
                        )
                    else:
                        yield self._format_data(
                            is_stream=False, model=model, content=content, usage=data.get("usage")
                        )

            except Exception as e:
                yield self._format_data(is_stream=body.get("stream", False), content=str(e))

        return StreamingResponse(stream_response())

    # ------------------------------------------------------------------ helpers
    def _extract_prompt_and_images(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, str]]]:
        prompt = ""
        images = []
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                prompt += content
                # crude inline base64 extraction
                inline = re.findall(r"!\[[^\]]*\]\(data:([^;]+);base64,([^)]+)\)", content)
                for mime, b64 in inline:
                    images.append({"mimeType": mime, "data": b64})
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        prompt += item.get("text", "")
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            header, encoded = url.split(",", 1)
                            mime = header.split(";")[0].split(":")[1]
                            images.append({"mimeType": mime, "data": encoded})
            break
        return prompt.strip(), images

    def _prep_file(self, img: Dict[str, str]) -> tuple:
        raw = base64.b64decode(img["data"])
        mime_type = img["mimeType"]
        ext = mime_type.split("/", 1)[-1]
        return (
            f"image.{ext}",
            raw,
            mime_type,
        )

    def _build_image_markup(self, data: Dict[str, Any]) -> List[str]:
        out = []
        for idx, img in enumerate(data.get("data", [])):
            if "b64_json" in img:
                src = f"data:image/png;base64,{img['b64_json']}"
            elif "url" in img:
                src = img["url"]
            else:
                continue
            out.append(f"![generated img {idx+1}]({src})")
        return out

    async def _upload_and_return_links(
        self, request: Request, user_dict: Dict, markup_list: List[str]
    ) -> List[str]:
        output = []
        user = Users.get_user_by_id(user_dict["id"])
        for mark in markup_list:
            # extract base64 again (cheap and dirty)
            m = re.search(r"base64,([^)]+)", mark)
            if not m:
                output.append(mark)
                continue
            b64 = m.group(1)
            raw = base64.b64decode(b64)
            upload = upload_file(
                request=request,
                background_tasks=BackgroundTasks(),
                file=UploadFile(
                    file=io.BytesIO(raw),
                    filename=f"{uuid.uuid4().hex}.png",
                    headers={"content-type": "image/png"},
                ),
                user=user,
                process=False,
            )
            url = request.app.url_path_for("get_file_content_by_id", id=upload.id)
            output.append(f"![generated]({url})")
        return output

    def _format_data(
        self,
        is_stream: bool,
        model: str = "",
        content: Optional[str] = "",
        usage: Optional[dict] = None,
    ) -> str:
        data = {
            "id": f"chat.{uuid.uuid4().hex}",
            "object": "chat.completion.chunk" if is_stream else "chat.completion",
            "created": int(time.time()),
            "model": model,
        }
        if content is not None:
            data["choices"] = [
                {
                    "index": 0,
                    "delta" if is_stream else "message": {"content": content},
                    "finish_reason": "stop" if not is_stream else None,
                }
            ]
        if usage:
            data["usage"] = usage
        return json.dumps(data) + "\0" if not is_stream else f"data: {json.dumps(data)}\n\n"