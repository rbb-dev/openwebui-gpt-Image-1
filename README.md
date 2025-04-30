# GPT-Image-1 for OpenWebUI

A clean, efficient plugin for generating and editing images using OpenAI's GPT-Image-1 model directly within OpenWebUI.

## Features

- Generate images from text prompts
- Edit existing images with text instructions
- Real-time status updates
- Support for multiple image sizes and quality settings
- Configurable moderation level

## Installation

1. In OpenWebUI, navigate to Plugins section
2. Click "Install from URL" or "Upload"
3. Add this plugin file
4. Configure your OpenAI API key in the plugin settings

## Configuration

The plugin has several configurable options:

- **OPENAI_API_KEY**: Your OpenAI API key (required)
- **IMAGE_COUNT**: Number of images to generate per request (1-10)
- **IMAGE_QUALITY**: Image quality (low, medium, high, auto)
- **IMAGE_SIZE**: Image size (1024x1024, 1536x1024, 1024x1536, auto)
- **MODERATION**: Moderation strictness (auto, low)

## Usage

### Generating Images

Simply type your prompt in the chat and select the GPT-Image-1 model from the model dropdown. The plugin will generate images based on your description.

Example:
```
A photorealistic sunset over mountains with a lake in the foreground
```

### Editing Images

To edit an image:
1. Send or upload an image in the chat
2. Add a text description of the edits you want
3. Select GPT-Image-1 model
4. Send the message

Example:
```
Change the sunset to sunrise and add birds flying in the sky
```
