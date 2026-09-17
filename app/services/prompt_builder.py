SYSTEM_PROMPT = """Tu es un assistant conversationnel utile et honnête.

- Réponds toujours en français, quelle que soit la langue de la question ou des documents fournis, sauf si l'utilisateur te demande explicitement de répondre dans une autre langue.
- Si des extraits de documents sont fournis dans un bloc "Contexte" ci-dessous, base ta réponse en priorité sur ces extraits et cite le fichier source (et la page si disponible) sous la forme [nom_fichier, page X]. N'invente jamais une information absente de ce contexte : si le contexte ne contient pas la réponse, dis-le clairement.
- S'il n'y a pas de bloc "Contexte", réponds normalement à la question avec tes connaissances générales, comme dans une conversation classique. Ne mentionne jamais de document ni de source dans ce cas.
"""


def _format_chunk(index: int, result) -> str:
    payload = result.payload or {}
    source = payload.get("source", "inconnu")
    page = payload.get("page")
    text = payload.get("text", "")
    header = f"[{index}] Source: {source}" + (f", page {page}" if page is not None else "")
    return f"{header}\n{text}"


def build_user_message(question: str, results: list) -> str:
    if not results:
        return question

    context_blocks = [_format_chunk(i + 1, result) for i, result in enumerate(results)]
    context = "\n\n".join(context_blocks)
    return f"""Contexte :
{context}

Question :
{question}
"""
