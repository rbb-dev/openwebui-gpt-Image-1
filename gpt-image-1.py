"""
title: GPT-Image-1
description: Simple plugin for generating and editing images using OpenAI's GPT-Image-1 model
author: Shayano
author_url: https://github.com/Shayano/
funding_url: https://github.com/open-webui
version: 1.0.0
"""

import base64
import json
import logging
import re
import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Awaitable, Literal

import httpx
from fastapi import Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse


class Pipe:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(
            default="", description="Your OpenAI API key"
        )
        API_BASE_URL: str = Field(
            default="https://api.openai.com/v1", 
            description="OpenAI API base URL"
        )
        IMAGE_COUNT: int = Field(
            default=1, 
            description="Number of images to generate (1-10)", 
            ge=1, 
            le=10
        )
        IMAGE_QUALITY: Literal["low", "medium", "high", "auto"] = Field(
            default="auto",
            description="Quality of the generated image"
        )
        IMAGE_SIZE: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = Field(
            default="1024x1024", 
            description="Size of the generated image"
        )
        MODERATION: Literal["auto", "low"] = Field(
            default="auto",
            description="Moderation strictness"
        )
        REQUEST_TIMEOUT: int = Field(
            default=600, 
            description="Request timeout in seconds"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.emitter: Optional[Callable[[dict], Awaitable[None]]] = None

    async def emit_status(self, message: str, done: bool = False, show_in_chat: bool = False):
        """Emit status updates to the client."""
        if self.emitter:
            await self.emitter({
                "type": "status",
                "data": {
                    "description": message,
                    "done": done
                }
            })
        
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
        
        async def stream_response():
            try:
                model = body.get("model", "gpt-image-1")
                messages = body.get("messages", [])
                is_stream = body.get("stream", False)
                
                # Extract prompt and images from messages
                prompt, images = self._extract_prompt_and_images(messages)
                
                if not self.valves.OPENAI_API_KEY:
                    yield self._format_data(
                        is_stream=is_stream,
                        content="Error: OpenAI API key not provided. Please set the OPENAI_API_KEY valve."
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
                async with httpx.AsyncClient(
                    base_url=self.valves.API_BASE_URL,
                    headers={"Authorization": f"Bearer {self.valves.OPENAI_API_KEY}"},
                    timeout=self.valves.REQUEST_TIMEOUT
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
                            images[0]["mimeType"]
                        )
                        
                        # Add other parameters
                        form_data.update({
                            "prompt": prompt,
                            "n": str(min(self.valves.IMAGE_COUNT, 10)),
                            "size": self.valves.IMAGE_SIZE if self.valves.IMAGE_SIZE != "auto" else "1024x1024",
                            "quality": self.valves.IMAGE_QUALITY if self.valves.IMAGE_QUALITY != "auto" else "standard",
                            "model": model.split(".", 1)[-1],
                        })
                        if self.valves.MODERATION != "auto":
                            form_data["moderation"] = self.valves.MODERATION
                        
                        await self.emit_status("🔄 Editing image... This may take a moment.")
                        response = await client.post(endpoint, data=form_data, files=files)
                    else:
                        # For image generations
                        json_data = {
                            "prompt": prompt,
                            "n": min(self.valves.IMAGE_COUNT, 10),
                            "size": self.valves.IMAGE_SIZE if self.valves.IMAGE_SIZE != "auto" else "1024x1024",
                            "quality": self.valves.IMAGE_QUALITY if self.valves.IMAGE_QUALITY != "auto" else "standard",
                            "model": model.split(".", 1)[-1],
                        }
                        if self.valves.MODERATION != "auto":
                            json_data["moderation"] = self.valves.MODERATION
                        
                        await self.emit_status("🔄 Generating image... This may take a moment.")
                        response = await client.post(endpoint, json=json_data)
                    
                    if response.status_code != 200:
                        error_message = f"Error from OpenAI API: {response.status_code} - {response.text}"
                        error_status = await self.emit_status(f"❌ An error occurred", True, True)
                        yield self._format_data(
                            is_stream=is_stream,
                            content=f"{error_status}{error_message}"
                        )
                        return
                    
                    response_data = response.json()
                    
                    # Process images - don't show this status in chat
                    await self.emit_status("✅ Image processing complete!", True)
                    
                    image_markdown = []
                    for i, img_data in enumerate(response_data.get("data", [])):
                        if "b64_json" in img_data:
                            image_markdown.append(
                                f"![image_{i+1}](data:image/png;base64,{img_data['b64_json']})"
                            )
                        elif "url" in img_data:
                            image_markdown.append(f"![image_{i+1}]({img_data['url']})")
                    
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
                            usage=response_data.get("usage")
                        )
                    else:
                        yield self._format_data(
                            is_stream=False,
                            model=model,
                            content=content,
                            usage=response_data.get("usage")
                        )
                    
            except Exception as err:
                error_status = await self.emit_status(f"❌ An error occurred", True, True)
                yield self._format_data(
                    is_stream=body.get("stream", False), 
                    content=f"{error_status}Error processing image request: {str(err)}"
                )

        return StreamingResponse(stream_response())

    def _extract_prompt_and_images(self, messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, str]]]:
        """Extract prompt and image data from messages."""
        prompt = ""
        images = []
        
        # Consider only the latest user message for simplicity
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
                
            content = message.get("content", "")
            
            # If content is a list (mixed content format)
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        prompt += item.get("text", "") + " "
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            parts = url.split(";base64,", 1)
                            if len(parts) == 2:
                                mime_type = parts[0].replace("data:", "", 1)
                                data = parts[1]
                                images.append({"mimeType": mime_type, "data": data})
            
            # If content is a string (look for embedded images)
            elif isinstance(content, str):
                prompt += content
                # Look for inline image markdown format: ![alt](data:mime/type;base64,DATA)
                pattern = r"!\[[^\]]*\]\(data:([^;]+);base64,([^)]+)\)"
                matches = re.findall(pattern, content)
                
                for mime_type, data in matches:
                    images.append({"mimeType": mime_type, "data": data})
                    # Remove the image part from the prompt
                    prompt = re.sub(pattern, "", prompt)
            
            # We only need the most recent user message
            break
        
        return prompt.strip(), images

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
