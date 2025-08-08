import os
import tempfile
import subprocess

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe_audio(file):
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(file.file.read())
        temp_audio_path = temp_audio.name

    # Compress to lower bitrate before sending
    compressed_path = temp_audio_path.replace(".webm", "_compressed.webm")
    subprocess.run(["ffmpeg", "-i", temp_audio_path, "-b:a", "32k", compressed_path, "-y"], check=True)

    # Send to Whisper
    with open(compressed_path, "rb") as audio:
        transcript = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=audio
        )

    # Delete temp files
    os.remove(temp_audio_path)
    os.remove(compressed_path)

    return transcript.text
