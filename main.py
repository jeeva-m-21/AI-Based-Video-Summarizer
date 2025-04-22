!pip install torch gradio pytubefix moviepy transformers huggingface_hub
import os
import torch
import gradio as gr
import pytubefix
from moviepy.editor import VideoFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from huggingface_hub import InferenceClient

# Initialize components
def initialize_components():
    # Initialize Whisper model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Initialize Inference Client
    client = InferenceClient(
        provider="fireworks-ai",
        api_key="hk_ ........",     # HuggingFace API Key
    )

    return pipe, client

# Function to download YouTube video
def download_video(video_url):
    try:
        video = pytubefix.YouTube(video_url)
        stream = video.streams.get_highest_resolution()
        download_path = "downloaded_video.mp4"
        stream.download(output_path=".", filename=download_path)
        return download_path, None
    except Exception as e:
        return None, f"Download error: {str(e)}"

# Convert video to audio
def video_to_audio(video_path, output_audio_path="output_audio.mp3"):
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(output_audio_path)
        return output_audio_path, None
    except Exception as e:
        return None, f"Audio conversion error: {str(e)}"

# Transcribe audio
def transcribe_audio(audio_path, pipe):
    try:
        result = pipe(audio_path, return_timestamps=True)
        return result["text"], None
    except Exception as e:
        return None, f"Transcription error: {str(e)}"

# Generate structured study notes
def generate_study_notes(transcript, client):
    try:
        prompt = f"""Transform this video transcript into comprehensive study notes with this exact structure:

# [Topic Name] Study Notes

## Key Concepts
- Bullet point 3-5 main ideas
- Include important definitions

## Detailed Breakdown
1. Main point 1 with supporting details
2. Main point 2 with supporting details
3. Main point 3 with supporting details

## Examples & Applications
- Practical examples mentioned
- Real-world applications

## Summary
Concise 3-4 sentence summary of core content

Transcript:
{transcript}"""

        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3  # Lower for more factual output
        )
        return completion.choices[0].message.content.strip(), None
    except Exception as e:
        return None, f"Notes generation error: {str(e)}"

# Main processing function
def process_video(youtube_url):
    # Initialize components
    pipe, client = initialize_components()

    # Step 1: Download video
    video_path, error = download_video(youtube_url)
    if error:
        return None, None, None, error

    # Step 2: Convert to audio
    audio_path, error = video_to_audio(video_path)
    if error:
        return video_path, None, None, error

    # Step 3: Transcribe audio
    transcript, error = transcribe_audio(audio_path, pipe)
    if error:
        return video_path, audio_path, None, error

    # Step 4: Generate study notes
    study_notes, error = generate_study_notes(transcript, client)
    if error:
        return video_path, audio_path, transcript, error

    return video_path, audio_path, transcript, study_notes

# Gradio Interface
with gr.Blocks(title="Video to Study Notes Converter", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎓 Video to Study Notes Converter
    Convert educational videos into structured study notes automatically
    """)

    with gr.Row():
        with gr.Column():
            youtube_url = gr.Textbox(
                label="YouTube Video URL",
                placeholder="Paste educational video URL here...",
                info="Works best with lectures, tutorials, and explainer videos"
            )
            process_btn = gr.Button("Generate Study Notes", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="Downloaded Video", interactive=False)
            audio_output = gr.Audio(label="Extracted Audio", visible=True)

    # ... (other code)

    with gr.Row():
       with gr.Column():
          transcript_output = gr.Textbox(
            label="Full Transcript",
            placeholder="Transcript will appear here...",
            lines=10,
            max_lines=20
        )

    with gr.Column():
        notes_output = gr.Markdown(  # Remove 'placeholder'
            label="Structured Study Notes",
            # placeholder="## Your notes will appear here...",  # Remove this line
        )
#... (rest of the code)

    error_output = gr.Textbox(label="Error Messages", visible=False)

    # Processing function
    def process_and_display(url):
        video, audio, transcript, notes = process_video(url)

        outputs = {
            video_output: video,
            audio_output: audio,
            transcript_output: transcript,
            notes_output: notes,
            error_output: gr.Textbox(visible=False)
        }

        if isinstance(notes, str) and notes.startswith("#"):
            outputs[notes_output] = notes  # Render as Markdown
        else:
            outputs[error_output] = gr.Textbox(value=notes if notes else "Unknown error", visible=True)

        return outputs

    process_btn.click(
        fn=process_and_display,
        inputs=[youtube_url],
        outputs=[video_output, audio_output, transcript_output, notes_output, error_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
