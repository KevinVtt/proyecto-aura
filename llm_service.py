import ollama

# Modelo de chat — corre 100% local, sin internet ni API key
CHAT_MODEL  = "gemma3"

# Modelo dedicado para embeddings — liviano (~274MB)
EMBED_MODEL = "nomic-embed-text"


def generar_embedding(texto: str) -> list[float]:
    """
    Convierte un texto en un vector numérico usando Ollama local.
    Requiere haber ejecutado: ollama pull nomic-embed-text
    """
    response = ollama.embeddings(model=EMBED_MODEL, prompt=texto)
    return response["embedding"]


def generar_respuesta(prompt: str) -> str:
    """
    Manda el prompt al LLM local y devuelve la respuesta.
    Requiere haber ejecutado: ollama pull gemma3
    """
    response = ollama.chat(
        model=CHAT_MODEL,
        options={"temperature": 0.2},
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
