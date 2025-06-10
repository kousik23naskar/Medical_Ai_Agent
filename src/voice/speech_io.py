import os
import requests
import json
from tempfile import NamedTemporaryFile
from configs.load_tools_config import LoadToolsConfig

# Load environment variables and configuration
tool_cfg = LoadToolsConfig()

# Voice → Text using Groq Whisper API
def transcribe_audio(audio_bytes: bytes) -> str:
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name

    whisper_url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"
    }
    files = {
        "file": (temp_path, open(temp_path, "rb")),
        "model": (None, tool_cfg.whisper_model),
        "temperature": (None, "0"),
        "response_format": (None, "verbose_json"),
        #"timestamp_granularities": (None, '["word"]'),
        "language": (None, "en")
    }

    response = requests.post(whisper_url, headers=headers, files=files)
    try:
        result = response.json()
        print("🔍 Whisper response JSON:", json.dumps(result, indent=2))
        if "text" in result:
            return result["text"]
        else:
            raise ValueError(f"No 'text' field in response: {result}")
    except Exception as e:
        return f"❌ Whisper transcription failed: {e}"

# Text → Voice using Groq TTS (playai-tts) — returns audio bytes
def synthesize_speech(text: str) -> bytes:
    tts_url = "https://api.groq.com/openai/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": tool_cfg.playai_model,  # e.g., "playai-tts"
        "input": text,
        "voice": tool_cfg.playai_voice,       #"Arista-PlayAI"  # Other options available
        "response_format": tool_cfg.playai_response_format,  # e.g., "mp3" or "wav"
    }

    response = requests.post(tts_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.content  # raw audio bytes
    else:
        print(f"❌ Groq TTS failed: {response.status_code} - {response.text}")
        return b""  # return empty bytes if failed

# Uncomment this block to use offline pyttsx3 TTS
# def synthesize_speech(text: str):
#     engine = pyttsx3.init()
#     engine.say(text)
#     engine.runAndWait()

# Uncomment this block to use OpenAI TTS
# def synthesize_speech(text: str):
#     response = openai.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text,
#     )
#     return response.content