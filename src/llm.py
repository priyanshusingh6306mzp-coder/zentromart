import os
import re # Add this import
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chat(messages, temperature=0.0):
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b", 
        messages=messages,
        temperature=temperature
    )
    raw_text = response.choices[0].message.content
    
    # Strip <think> tags if the model generates them
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    return clean_text