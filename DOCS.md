# Comprendre le RAG et le fonctionnement de `rag-api`

Ce document explique, en partant de zéro, ce qu'est un RAG, pourquoi on en a besoin, puis détaille précisément comment ce backend l'implémente : chaque étape, chaque fichier, chaque choix algorithmique.

Pour l'installation et la référence des endpoints, voir [README.md](README.md). Ce document-ci répond à la question « comment ça marche, en interne ? ».

---

## 1. C'est quoi, le RAG ?

**RAG** = *Retrieval-Augmented Generation* (génération augmentée par récupération).

Un LLM (modèle de langage) seul a deux limites structurelles :

1. **Connaissances figées** : il ne connaît que ce qu'il a vu à l'entraînement. Il ne peut pas répondre sur *tes* documents (un contrat, un rapport, des notes personnelles) qu'il n'a jamais vus.
2. **Hallucinations** : quand il ne sait pas, il a tendance à inventer une réponse plausible plutôt que d'admettre son ignorance.

Le RAG répond aux deux en changeant la façon de poser la question au modèle. Au lieu de demander directement « Réponds à X », on fait en deux temps :

1. **Retrieval (récupération)** : on cherche, parmi une base de documents, les passages les plus pertinents par rapport à la question.
2. **Augmented Generation (génération augmentée)** : on donne ces passages au LLM *avec* la question, en lui disant « réponds en te basant sur ce contexte ».

Le LLM ne « sait » toujours rien de plus qu'avant — mais on lui fournit la bonne information au bon moment, dans sa fenêtre de contexte, et on lui demande de citer ses sources plutôt que d'inventer.

### Comment sait-on quels passages sont « pertinents » ?

C'est le cœur technique du RAG : les **embeddings**.

Un embedding est un vecteur de nombres (ex. 768 dimensions) qui représente le *sens* d'un texte. Deux textes proches en sens ont des vecteurs proches dans cet espace — on mesure cette proximité avec la **similarité cosinus** (un score entre -1 et 1, où 1 = sens identique).

Le principe :

- **À l'indexation** : chaque document est découpé en petits morceaux (*chunks*), et chaque chunk est transformé en vecteur (embedding), stocké dans une base vectorielle.
- **À la question** : la question elle-même est transformée en vecteur avec le même modèle, puis on cherche dans la base les chunks dont le vecteur est le plus proche — c'est une recherche par similarité, pas une recherche par mot-clé.

C'est exactement ce que fait ce projet, avec :

- [**Ollama**](https://ollama.com) qui fait tourner localement le modèle d'embeddings (`nomic-embed-text`) et le modèle de chat (`qwen3:4b` par défaut) ;
- [**Qdrant**](https://qdrant.tech) qui stocke les vecteurs et fait la recherche par similarité.

Tout tourne en local : aucune donnée ne part vers une API externe.

---

## 2. Vue d'ensemble de l'architecture

```
┌───────────┐        HTTP / NDJSON        ┌───────────┐
│  rag-ui   │ ───────────────────────────▶│  rag-api  │
│ (React)   │◀─────────────────────────── │ (FastAPI) │
└───────────┘                              └─────┬─────┘
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                          ▼                       ▼                       │
                   ┌─────────────┐         ┌─────────────┐               │
                   │   Ollama    │         │   Qdrant    │               │
                   │ (embeddings │         │  (vecteurs, │               │
                   │  + chat LLM)│         │  recherche) │               │
                   └─────────────┘         └─────────────┘               │
                                                                          │
                                            fichiers uploadés ───────────┘
                                            (data/uploads/)
```

Deux flux distincts traversent ce backend :

- **Le flux d'indexation** (quand on upload un document) — décrit en [§3](#3-pipeline-dindexation-dun-document).
- **Le flux de question/réponse** (quand on pose une question) — décrit en [§4](#4-pipeline-de-questionréponse).

---

## 3. Pipeline d'indexation d'un document

Fichier d'entrée : [`app/routers/documents.py`](app/routers/documents.py), route `POST /documents/upload`.

```
Fichier (pdf/docx/txt/md)
        │
        ▼
 [1] Upload ─────────────────── routers/documents.py
        │  sauvegarde brute sur disque (UPLOAD_DIR)
        ▼
 [2] Parsing ─────────────────── services/parsers.py
        │  texte brut → liste de "Section" (texte + page/titre)
        ▼
 [3] Chunking ─────────────────── services/chunker.py
        │  chaque Section → une ou plusieurs "Chunk" (fenêtre glissante)
        ▼
 [4] Embedding ─────────────────── services/embeddings.py
        │  chaque Chunk → un vecteur, via Ollama
        ▼
 [5] Stockage ─────────────────── services/qdrant_store.py
        │  upsert des vecteurs + métadonnées dans Qdrant
        ▼
   Document interrogeable
```

L'orchestration de ces étapes se fait dans [`services/ingestion.py`](app/services/ingestion.py), fonction `ingest_file`.

### 3.1. Upload

`routers/documents.py` reçoit le fichier, en extrait le nom (`Path(file.filename).name`, pour éviter tout chemin type `../../etc/passwd`), et l'écrit tel quel dans `UPLOAD_DIR` (`./data/uploads` par défaut). Le nom de fichier sert d'**identifiant unique du document** (`source`) pour tout le reste du pipeline.

### 3.2. Parsing — extraire le texte selon le type de fichier

`services/parsers.py` transforme chaque type de fichier en une liste de `Section` :

```python
@dataclass
class Section:
    text: str
    page: int | None = None
    heading: str | None = None
```

| Type | Découpage en `Section` | Détail |
|---|---|---|
| `.pdf` | **une section par page** | `page` = numéro de page. Les pages sans texte extractible (scan sans OCR) sont ignorées. |
| `.docx` | **une section par titre** (`Heading 1..N`) | Le texte entre deux titres forme une section, avec `heading` renseigné. S'il n'y a aucun titre dans le document, tout le texte est mis dans une seule section (fallback). |
| `.txt` | **une seule section pour tout le fichier** | Les paragraphes (séparés par des lignes vides) sont simplement recollés. |
| `.md` / `.markdown` | **une section par titre Markdown** (`#`, `##`, ...) | Même logique que le `.docx`, avec fallback si aucun titre. |

**Pourquoi c'est important pour la suite** : le chunking (étape suivante) s'applique *indépendamment à chaque section*. Concrètement :

- Pour un PDF, **un chunk ne traverse jamais deux pages** — ça permet de toujours savoir de quelle page vient un passage cité.
- Pour un DOCX/Markdown, **un chunk ne traverse jamais deux sections de titre**.
- Pour un `.txt`, comme tout le fichier est une seule section, **un chunk peut regrouper plusieurs paragraphes**.

### 3.3. Chunking — découper en morceaux de taille gérable

Fichier : [`services/chunker.py`](app/services/chunker.py).

**Pourquoi découper ?** Un embedding représente le sens d'un texte *global* : si on encode un document entier de 10 pages en un seul vecteur, on perd toute la granularité (la question porte souvent sur un point précis, pas sur l'ensemble). On découpe donc en chunks assez petits pour rester spécifiques, mais assez grands pour garder du contexte.

**L'algorithme — fenêtre glissante avec chevauchement :**

```python
step = max(chunk_size_words - overlap_words, 1)

start = 0
while start < len(words):
    window = words[start : start + chunk_size_words]
    # ... créer un chunk avec `window`
    if start + chunk_size_words >= len(words):
        break
    start += step
```

Avec les valeurs par défaut (`CHUNK_SIZE_WORDS=480`, `CHUNK_OVERLAP_WORDS=75`), le pas d'avancement est `step = 480 - 75 = 405` mots.

**Exemple concret** avec une section de 1000 mots :

| Chunk | Mots couverts | Taille |
|---|---|---|
| 0 | 0 → 480 | 480 mots |
| 1 | 405 → 885 | 480 mots |
| 2 | 810 → 1000 | 190 mots (dernier chunk, tronqué) |

Le chunk 1 partage exactement 75 mots avec le chunk 0 (480 - 405 = 75), et 75 mots avec le chunk 2 (885 - 810 = 75) — c'est voulu.

**Pourquoi chevaucher les chunks ?** Sans chevauchement, une idée à cheval sur la fin d'un chunk et le début du suivant serait coupée en deux et perdrait son sens dans chacun des deux morceaux. Le chevauchement garantit qu'une phrase-clé proche d'une frontière apparaît intacte dans au moins un chunk.

Si une section fait moins de `CHUNK_SIZE_WORDS` mots, elle devient un chunk unique, sans découpage.

### 3.4. Embedding — vectoriser chaque chunk

Fichier : [`services/embeddings.py`](app/services/embeddings.py).

Chaque chunk est envoyé à Ollama (modèle `nomic-embed-text` par défaut) pour obtenir son vecteur. Deux détails importants :

**a) Préfixes asymétriques.** `nomic-embed-text` (comme beaucoup de modèles d'embeddings modernes) est entraîné avec des préfixes différents selon qu'on encode un document ou une requête :

```python
def embed_document_chunks(self, texts):
    prefixed = [f"search_document: {text}" for text in texts]
    ...

def embed_query(self, text):
    vectors = self.embed_texts([f"search_query: {text}"])
    ...
```

Une question (« Quel est le délai de préavis ? ») et la phrase qui y répond (« Le délai de préavis est de 3 mois. ») ne sont pas formulées pareil, même si elles parlent de la même chose. Le préfixe indique au modèle *dans quel rôle* il encode le texte, ce qui améliore nettement la qualité du rapprochement question ↔ passage. **Utiliser le mauvais préfixe (ou aucun) dégrade la pertinence de la recherche.**

**b) Traitement par lot.** `services/ingestion.py` envoie les chunks à embedder par paquets de 8 (`BATCH_SIZE = 8`), pour limiter le nombre d'appels réseau à Ollama.

### 3.5. Stockage — indexation dans Qdrant

Fichier : [`services/qdrant_store.py`](app/services/qdrant_store.py).

Avant le tout premier upload, la collection Qdrant n'existe pas. `ensure_collection` la crée à la volée, avec une taille de vecteur déterminée dynamiquement (`embeddings.get_dimension()` fait un appel de sondage à Ollama pour connaître la dimension du modèle configuré) et une distance cosinus :

```python
VectorParams(size=vector_size, distance=Distance.COSINE)
```

Pour chaque chunk, un point est créé avec :

- un **ID déterministe** : `uuid5(NAMESPACE_URL, f"{source}:{chunk_index}")` — le même fichier avec le même chunk génère toujours le même UUID. Ce n'est pas un hasard : cela rend le ré-upload idempotent.
- un **payload** (métadonnées) : `source` (nom du fichier), `file_type`, `page`, `heading`, `chunk_index`, `text` (le texte du chunk, nécessaire pour construire le contexte plus tard), `indexed_at`.

**Ré-indexation d'un document existant.** `ingest_file` appelle `store.delete_by_source(source)` *avant* d'insérer les nouveaux chunks. Donc réuploader un fichier du même nom remplace intégralement son ancienne version dans l'index (pas d'accumulation de doublons, pas de chunks fantômes si le nouveau fichier a moins de chunks que l'ancien).

---

## 4. Pipeline de question/réponse

Fichier d'entrée : [`app/routers/chat.py`](app/routers/chat.py), route `POST /chat/ask`.

```
Question + historique de conversation
        │
        ▼
 [1] Embedding de la question ── "search_query: <question>"
        ▼
 [2] Recherche vectorielle ────── top_k chunks les plus proches (Qdrant, cosinus)
        ▼
 [3] Filtrage ───────────────────  ne garde que score ≥ SIMILARITY_THRESHOLD
        ▼
 [4] Troncature de l'historique ── garde les MAX_HISTORY_TURNS derniers tours
        ▼
 [5] Construction du prompt ────── system prompt + historique + contexte + question
        ▼
 [6] Génération streamée ───────── Ollama, séparation "thinking" / "content"
        ▼
 [7] Flux NDJSON vers le client ── sources → thinking → tokens → done
```

### 4.1. Recherche des passages pertinents

```python
query_vector = embeddings.embed_query(request.question)
all_results = store.search(query_vector, top_k)
relevant_results = [r for r in all_results if r.score >= settings.similarity_threshold]
```

`store.search` interroge Qdrant pour les `top_k` chunks les plus proches du vecteur de la question (par défaut `TOP_K=5`), **sans filtrage de score côté Qdrant** — Qdrant renvoie toujours ses `top_k` meilleurs résultats, même s'ils sont peu pertinents (par exemple si la base est presque vide, ou si la question n'a aucun rapport avec les documents indexés).

Le filtrage par `SIMILARITY_THRESHOLD` (0.65 par défaut) se fait donc **côté application**, juste après. C'est ce filtre qui décide si le LLM aura du contexte documentaire ou non pour cette question.

### 4.2. Le cas « pas de contexte pertinent »

Si aucun résultat ne dépasse le seuil, `relevant_results` est vide. `prompt_builder.build_user_message` le détecte et **ne construit aucun bloc « Contexte »** :

```python
def build_user_message(question, results):
    if not results:
        return question
    ...
```

Combiné à cette règle du system prompt :

> S'il n'y a pas de bloc "Contexte", réponds normalement à la question avec tes connaissances générales... Ne mentionne jamais de document ni de source dans ce cas.

Le système bascule ainsi naturellement en **chat généraliste** quand la question ne concerne pas les documents indexés, plutôt que de forcer une citation de sources non pertinentes. C'est ce qui permet à l'interface de servir à la fois de RAG *et* de chatbot classique.

### 4.3. Construction du prompt

Fichier : [`services/prompt_builder.py`](app/services/prompt_builder.py).

Quand il y a du contexte, chaque chunk retenu est formaté ainsi :

```
[1] Source: contrat.pdf, page 3
<texte du chunk>

[2] Source: notes.md
<texte du chunk>
```

Puis englobé dans le message utilisateur final :

```
Contexte :
[1] Source: contrat.pdf, page 3
...

[2] Source: notes.md
...

Question :
<la question posée>
```

Le system prompt demande explicitement au modèle de citer ses sources sous la forme `[nom_fichier, page X]` et de ne jamais inventer une information absente du contexte.

### 4.4. Historique de conversation

```python
history = request.history[-(settings.max_history_turns * 2):]
```

Le frontend envoie tout l'historique de la session à chaque question. Le backend ne garde que les `MAX_HISTORY_TURNS` derniers tours (par défaut 6, soit 12 messages : 6 côté utilisateur + 6 côté assistant) avant de les injecter dans le prompt envoyé au LLM. Cela borne la taille du contexte envoyé au modèle, indépendamment de la longueur réelle de la conversation.

### 4.5. Génération streamée

Fichier : [`services/llm.py`](app/services/llm.py).

```python
for chunk in client.chat(model=model, messages=messages, stream=True):
    if chunk.message.thinking:
        yield {"event": "thinking", "text": chunk.message.thinking}
    elif chunk.message.content:
        yield {"event": "token", "text": chunk.message.content}
yield {"event": "done"}
```

Ollama renvoie les réponses token par token. Certains modèles (comme `qwen3`, configuré par défaut) exposent un champ `thinking` séparé pour leur raisonnement interne avant la réponse finale — le code distingue les deux flux, ce qui permet au frontend d'afficher la réflexion dans un bloc repliable, séparé de la réponse.

### 4.6. Le flux NDJSON

`routers/chat.py` renvoie une `StreamingResponse` en `application/x-ndjson` : une ligne JSON par événement, envoyée au fil de l'eau (pas de buffering complet côté serveur) :

```jsonc
{"type": "sources", "sources": [...]}      // envoyé immédiatement, avant la génération
{"type": "thinking", "content": "..."}      // 0 à N fois
{"type": "token", "content": "..."}          // 0 à N fois
{"type": "done"}                              // fin normale
{"type": "error", "message": "..."}            // en cas d'erreur pendant la génération
```

Les sources sont envoyées **avant** le premier token de réponse : le frontend peut donc afficher « Sources » dès le début, sans attendre la fin de la génération.

### 4.7. Gestion des erreurs — une asymétrie à connaître

Il y a une différence de comportement selon *quand* l'erreur survient :

- **Avant le stream** (embedding de la question, recherche Qdrant) : ces appels sont faits en dehors du générateur, de façon synchrone. Une exception ici (Ollama ou Qdrant injoignable) **n'est pas interceptée** et remonte comme une erreur HTTP 500 classique.
- **Pendant le stream** (génération de la réponse) : le code est dans un `try/except` qui transforme toute exception en événement `{"type": "error", ...}` dans le flux NDJSON — car à ce stade, les en-têtes HTTP 200 sont déjà envoyés, il est trop tard pour changer le code de statut.

Le frontend gère ce deuxième cas : en recevant un événement `error`, il lève une exception JS qui est affichée comme message d'erreur dans la conversation.

---

## 5. Les autres endpoints

- **`GET /health`** ([`main.py`](app/main.py)) : fait un vrai appel à Ollama (`embed_texts(["ping"])`) et à Qdrant (`get_collections()`) pour vérifier qu'ils répondent réellement — ce n'est pas un simple "l'API est démarrée", mais des sondes actives sur les deux dépendances.
- **`GET /models`** ([`services/ollama_models.py`](app/services/ollama_models.py)) : interroge `GET /api/tags` sur Ollama et ne garde que les modèles qui déclarent la capacité `"completion"` (donc utilisables pour le chat — les modèles purement embeddings, comme `nomic-embed-text`, sont exclus de cette liste).
- **`GET /documents`** / **`DELETE /documents/{source}`** ([`services/qdrant_store.py`](app/services/qdrant_store.py), `list_sources` / `delete_by_source`) : `list_sources` parcourt (`scroll`) tous les points de la collection par pages de 256 et les agrège par `source` pour reconstruire la liste des documents et leur nombre de chunks — il n'existe pas de table « documents » séparée, tout est dérivé des payloads des chunks eux-mêmes.

---

## 6. Impact des variables de configuration sur l'algorithme

| Variable | Effet algorithmique |
|---|---|
| `CHUNK_SIZE_WORDS` | Taille de la fenêtre de chunking. Plus grand = plus de contexte par chunk mais embeddings moins spécifiques. |
| `CHUNK_OVERLAP_WORDS` | Chevauchement entre chunks consécutifs. Plus grand = moins de risque de couper une idée en deux, mais plus de chunks (donc plus de stockage et d'appels d'embedding) pour un même document. |
| `TOP_K` | Nombre de chunks candidats remontés par Qdrant *avant* filtrage par seuil. |
| `SIMILARITY_THRESHOLD` | Seuil de cosinus en dessous duquel un chunk est ignoré. Trop bas = bruit dans le contexte (risque d'hallucination guidée par du contenu non pertinent). Trop haut = bascule trop souvent en mode « chat généraliste » alors que la réponse existait dans les documents. |
| `MAX_HISTORY_TURNS` | Nombre de tours d'historique conservés dans le prompt envoyé au LLM — ne réduit pas la mémoire de session côté frontend, seulement ce qui est transmis au modèle. |

---

## 7. Limites connues de l'implémentation actuelle

- **Chunking par mots, pas par tokens** : le découpage compte des mots (`str.split()`), pas des tokens du modèle. Pour des textes avec beaucoup de ponctuation ou de caractères non-latins, la taille réelle en tokens envoyée au LLM peut varier sensiblement par rapport à `CHUNK_SIZE_WORDS`.
- **Pas de reranking** : les chunks remontés par la recherche vectorielle sont utilisés tels quels, sans étape de ré-évaluation plus fine (cross-encoder) qui améliorerait la précision du top-k.
- **Un seul type de recherche** : uniquement de la recherche vectorielle dense (pas de recherche lexicale/BM25 hybride), donc potentiellement moins bon sur des requêtes très précises (numéros de référence, noms propres exacts) que sur des requêtes sémantiques.
- **Collection Qdrant unique et globale** : tous les documents de tous les utilisateurs partagent la même collection (`QDRANT_COLLECTION`) — pas d'isolation multi-utilisateur.
- **Pas de suppression globale côté API** : voir la note dans [README.md](README.md#limites-connues).
