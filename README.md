# Headshot

An AI-powered professional headshot generator. Upload a selfie, adjust the style, and get a polished professional photo in seconds.

## What It Does

1. Upload or take a photo
2. Crop to frame your face
3. Customise the output with prompt controls (style, background, lighting)
4. The AI generates a professional headshot via a Hugging Face Spaces model
5. Download the result

Includes a 20-generation limit per session to manage API usage.

## Tech Stack

- **Backend**: Flask
- **AI Model**: Hosted on Hugging Face Spaces (Gradio client)
- **Image Processing**: Pillow
- **Frontend**: HTML/CSS/JS with canvas-based cropping
- **Deployment**: Render (render.yaml included)

## Setup

```bash
git clone https://github.com/sedici16/headshot.git
cd headshot
pip install -r requirements.txt
```

Create a `.env` file:

```
HF_TOKEN=your-huggingface-token
```

Run:

```bash
python app.py
```

## Project Structure

```
headshot/
├── app.py              # Flask app, handles uploads and AI generation
├── requirements.txt
├── render.yaml         # Render deployment config
├── static/             # CSS, JS, images
└── templates/          # HTML templates (landing page + form)
```
