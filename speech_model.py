from groq import Groq
from gtts import gTTS

# Initialize Groq client (replace with your API key)

import os
from dotenv import load_dotenv
 
load_dotenv()

client = Groq(api_key=os.getenv("GROK_SPEECH"))

    
def create_prompt(text):
    return f"""
You are an AI voice assistant for air quality alerts.

Convert the following AQI analysis into a natural spoken alert.

Rules:
- Simple English
- Max 2-3 sentences
- Add warning tone
- Include health advice

AQI Data:
{text}

Voice Alert:
"""


def generate_speech_text():
    # Read AQI explanation
    with open("explanation.txt", "r") as file:
        aqi_text = file.read()

    # Generate alert text using LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": create_prompt(aqi_text)}
        ]
    )

    speech_text = response.choices[0].message.content

    return speech_text


def text_to_speech(speech_text):
    tts = gTTS(speech_text)
    tts.save("alert.mp3")
    print("Audio saved as alert.mp3")


def main():
    speech_text = generate_speech_text()
    text_to_speech(speech_text)


if __name__ == "__main__":
    main()