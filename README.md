# RAG API

API backend en Python/FastAPI pour un système de RAG (Retrieval-Augmented Generation) entièrement local : les modèles tournent via [Ollama](https://ollama.com) et les embeddings sont stockés dans [Qdrant](https://qdrant.tech). Aucune donnée ne sort de la machine.

Elle sert de backend au frontend [`rag-ui`](../rag-ui).

## Fonctionnalités

- **Ingestion de documents** : upload de fichiers `.pdf`, `.docx`, `.txt`, `.md`/`.markdown`, découpage en chunks et indexation vectorielle dans Qdrant.
- **Chat streamé** : endpoint de question/réponse qui renvoie un flux NDJSON (sources → tokens de réflexion → tokens de réponse), avec historique de conversation.
- **Recherche sémantique** avec seuil de similarité configurable, injectée comme contexte au modèle de chat.
- **Citation des sources** : chaque réponse est accompagnée des documents/pages utilisés comme contexte.
- **Gestion documentaire** : liste, suppression par document.
- **Sélection de modèle** : liste des modèles de chat disponibles sur l'instance Ollama.
- **Healthcheck** de la disponibilité d'Ollama et de Qdrant.

## Architecture

```
app/
├── main.py                  # Point d'entrée FastAPI, CORS, routes, healthcheck
├── config.py                 # Configuration via variables d'environnement (.env)
├── deps.py                    # Instanciation des dépendances partagées (embeddings, store)
├── schemas.py                 # Modèles Pydantic (requêtes/réponses)
├── routers/
│   ├── documents.py           # Upload, liste, suppression de documents
│   ├── chat.py                 # Endpoint de question/réponse streamé
│   └── models.py               # Liste des modèles de chat Ollama disponibles
└── services/
    ├── parsers.py               # Extraction de texte par type de fichier (pdf/docx/txt/md)
    ├── chunker.py                # Découpage du texte en chunks avec overlap
    ├── embeddings.py             # Client d'embeddings Ollama (nomic-embed-text par défaut)
    ├── qdrant_store.py            # Client Qdrant : collection, upsert, recherche, suppression
    ├── ingestion.py                # Orchestration parse → chunk → embed → upsert
    ├── prompt_builder.py           # Construction du system prompt et du contexte RAG
    └── ollama_models.py             # Récupération des modèles de chat disponibles
```

## Prérequis

- Python 3.11+
- [Ollama](https://ollama.com) démarré localement, avec les modèles nécessaires téléchargés :
  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen3:4b
  ```
- [Qdrant](https://qdrant.tech) accessible (par exemple via Docker) :
  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # puis ajuster si besoin
```

## Lancement

```bash
uvicorn app.main:app --reload --port 8000
```

L'API est alors disponible sur `http://localhost:8000`, avec la doc interactive sur `http://localhost:8000/docs`.

## Configuration (`.env`)

| Variable | Défaut | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL de l'instance Ollama |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Modèle utilisé pour générer les embeddings |
| `OLLAMA_CHAT_MODEL` | `qwen3:4b` | Modèle de chat par défaut |
| `QDRANT_HOST` | `127.0.0.1` | Hôte Qdrant |
| `QDRANT_PORT` | `6333` | Port Qdrant |
| `QDRANT_COLLECTION` | `documents` | Nom de la collection vectorielle |
| `CHUNK_SIZE_WORDS` | `480` | Taille maximale d'un chunk, en mots |
| `CHUNK_OVERLAP_WORDS` | `75` | Chevauchement entre chunks consécutifs, en mots |
| `TOP_K` | `5` | Nombre de chunks remontés par recherche |
| `SIMILARITY_THRESHOLD` | `0.65` | Score de similarité cosinus minimum pour qu'un chunk soit retenu comme contexte |
| `MAX_HISTORY_TURNS` | `6` | Nombre de tours de conversation conservés dans l'historique envoyé au modèle |
| `UPLOAD_DIR` | `./data/uploads` | Dossier de stockage des fichiers uploadés |
| `CORS_ORIGINS` | `http://localhost:5173` | Origines autorisées (liste séparée par des virgules) |

## Endpoints

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/health` | Statut de l'API et disponibilité d'Ollama / Qdrant |
| `GET` | `/documents` | Liste des documents indexés (source, type, nombre de chunks, date d'indexation) |
| `POST` | `/documents/upload` | Upload et indexation d'un fichier (`multipart/form-data`, champ `file`) |
| `DELETE` | `/documents/{source}` | Suppression d'un document et de tous ses chunks |
| `POST` | `/chat/ask` | Question/réponse streamée (NDJSON) avec `question`, `history`, `top_k?`, `model?` |
| `GET` | `/models` | Liste des modèles de chat disponibles sur Ollama + modèle par défaut |

### Format du flux `/chat/ask`

Chaque ligne de la réponse est un objet JSON :

```jsonc
{"type": "sources", "sources": [...]}
{"type": "thinking", "content": "..."}   // tokens de réflexion (modèles reasoning)
{"type": "token", "content": "..."}       // tokens de la réponse
{"type": "done"}
{"type": "error", "message": "..."}
```

## Limites connues

- Aucune route de réinitialisation globale (`DELETE /documents` sans source) n'est exposée côté API ; la suppression se fait document par document via `DELETE /documents/{source}`.
