import os
from dotenv import load_dotenv
from groq import Groq
import ollama

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

CHAT_MODEL  = "llama-3.3-70b-versatile"  # el mejor en Groq free tier
EMBED_MODEL = "nomic-embed-text"          # Ollama local para embeddings


def generar_embedding(texto: str) -> list[float]:
    """Embeddings con Ollama local."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=texto)
    return response["embedding"]


def generar_respuesta(prompt: str) -> str:
    """Respuestas con Groq — responde en ~1-2 segundos."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content