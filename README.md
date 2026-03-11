# Aura Chatbot — Guía con Ollama (100% local, sin API key)

## ¿Por qué Ollama?
Todo corre en tu PC. Ningún dato de la empresa sale a internet.
No necesitás pagar ninguna API ni crear cuentas externas.

---

## Estructura del proyecto

```
aura_chatbot/
├── main.py                  ← ChatbotController (API REST)
├── chat_service.py          ← ChatService (lógica del chat + historial)
├── document_processor.py    ← DocumentProcessor (chunking + embeddings)
├── llm_service.py           ← LLMService (Ollama local)
├── requirements.txt
├── pdfs/                    ← Los PDFs que subas se guardan acá
└── chroma_db/               ← ChromaDB guarda los vectores acá (se crea solo)
```

---

## PASO 1 — Instalar Ollama

1. Entrá a https://ollama.com/download
2. Descargá el instalador para Windows y ejecutalo
3. Al terminar, Ollama queda corriendo como servicio en segundo plano

Verificá en CMD:
```
ollama --version
```

---

## PASO 2 — Descargar los modelos (una sola vez)

```bash
ollama pull nomic-embed-text
```
Modelo de embeddings (~274MB) — convierte texto en vectores para ChromaDB.

```bash
ollama pull gemma3
```
Modelo de chat (~3.3GB) — genera las respuestas del chatbot.

Verificá que quedaron:
```bash
ollama list
```

---

## PASO 3 — Instalar dependencias Python

```bash
pip install -r requirements.txt
```

---

## PASO 4 — Levantar el servidor

```bash
python -m uvicorn main:app --reload
```

---

## PASO 5 — Usar el chatbot

Abrí: http://127.0.0.1:8000/docs

1. POST /documentos/subir → subís tu PDF
2. POST /sesion/nueva → copiás el sesion_id
3. POST /chat → hacés preguntas con ese sesion_id
4. Seguí mandando preguntas con el mismo sesion_id para mantener el historial

---

## Requisitos de hardware

| RAM  | Disco libre |
|------|-------------|
| 8 GB mínimo | 5 GB mínimo |

Con GPU Nvidia responde más rápido. Sin GPU tarda ~10-30 seg por respuesta.
Para respuestas más rápidas cambiá "gemma3" por "gemma3:1b" en llm_service.py

---

## Solución de problemas

"connection refused" → Ollama no está corriendo, abrilo desde el menú inicio.
"model not found"    → Ejecutá ollama pull gemma3 y ollama pull nomic-embed-text.
Respuestas lentas    → Normal sin GPU. Usá gemma3:1b para mayor velocidad.
