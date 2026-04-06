from groq import Groq
import os
from dotenv import load_dotenv
 
load_dotenv()
client = Groq(api_key= os.getenv("GROK_API"))


def generate_explanation(aqi, pm25, pm10, no2):
    prompt = f"""
    You are an expert in air quality analysis.

    AQI level is {aqi}.
    Pollutant concentrations are:
    - PM2.5: {pm25}
    - PM10: {pm10}
    - NO2: {no2}

    Explain the situation in a detailed paragraph including:
    - Main causes of pollution
    - Health effects on humans
    - Precautions and safety measures
    - Possible sources (traffic, industry, dust, etc.)
    - Suggestions to improve air quality

    Provide a clear, informative, and human-readable explanation.
    """
    return prompt


def main():
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": generate_explanation(185, 70, 100, 50)}
        ]
    )

    model_text = response.choices[0].message.content

    print(model_text)

    # Save output to file
    with open("explanation.txt", "w") as f:
        f.write(model_text)

    print("Explanation saved to explanation.txt")


if __name__ == "__main__":
    main()