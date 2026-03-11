from document_processor import buscar_chunks_relevantes
from llm_service import generar_respuesta

# Historial en memoria: { sesion_id: [ {rol, contenido}, ... ] }
_historiales: dict[str, list[dict]] = {}

UMBRAL_CHUNKS = 3  # mínimo de chunks para considerar que hay contexto


def _construir_prompt(pregunta: str, chunks: list[dict], historial: list[dict]) -> str:
    """
    Arma el prompt final que se manda al LLM con:
    - Instrucción de comportamiento
    - Contexto recuperado de los PDFs
    - Historial de la conversación
    - Pregunta actual
    """

    # --- Contexto de documentos ---
    contexto = ""
    for i, chunk in enumerate(chunks, 1):
        contexto += f"\n[Fragmento {i} - Fuente: {chunk['fuente']}]\n{chunk['texto']}\n"

    # --- Historial de conversación ---
    historial_texto = ""
    for msg in historial:
        rol = "Usuario" if msg["rol"] == "user" else "Asistente"
        historial_texto += f"{rol}: {msg['contenido']}\n"

    prompt = f"""Sos un asistente de la empresa. Tu trabajo es responder preguntas
usando ÚNICAMENTE la información que aparece en los fragmentos de documentos provistos.

Si la respuesta no está en los fragmentos, decí exactamente:
"No encontré información sobre ese tema en los documentos disponibles."

No inventes información. No uses conocimiento externo.

=== DOCUMENTOS DE LA EMPRESA ===
{contexto}

=== HISTORIAL DE CONVERSACIÓN ===
{historial_texto if historial_texto else "(sin conversación previa)"}

=== PREGUNTA ACTUAL ===
{pregunta}

=== TU RESPUESTA ==="""

    return prompt


def responder(sesion_id: str, pregunta: str) -> dict:
    """
    Flujo completo:
    1. Recupera historial de la sesión
    2. Busca chunks relevantes en ChromaDB
    3. Construye el prompt con contexto + historial
    4. Llama al LLM
    5. Guarda la interacción en el historial
    6. Devuelve respuesta + fuentes usadas
    """

    # 1. Obtener o crear historial de esta sesión
    if sesion_id not in _historiales:
        _historiales[sesion_id] = []
    historial = _historiales[sesion_id]

    # 2. Buscar contexto relevante en los PDFs
    chunks = buscar_chunks_relevantes(pregunta, top_k=4)

    # 3. Construir prompt
    prompt = _construir_prompt(pregunta, chunks, historial)

    # 4. Llamar al LLM
    respuesta = generar_respuesta(prompt)

    # 5. Guardar en historial (solo los últimos 10 turnos para no crecer infinito)
    historial.append({"rol": "user",      "contenido": pregunta})
    historial.append({"rol": "assistant", "contenido": respuesta})
    if len(historial) > 20:  # 10 turnos × 2 mensajes
        historial = historial[-20:]
    _historiales[sesion_id] = historial

    # 6. Devolver
    fuentes = list(set(c["fuente"] for c in chunks))
    return {
        "respuesta": respuesta,
        "fuentes":   fuentes,
        "historial": historial
    }


def obtener_historial(sesion_id: str) -> list[dict]:
    return _historiales.get(sesion_id, [])


def limpiar_historial(sesion_id: str):
    _historiales.pop(sesion_id, None)
