import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Available models on your API key:")
print("─" * 40)
for model in client.models.list():
    if "gemini" in model.name.lower():
        print(model.name)