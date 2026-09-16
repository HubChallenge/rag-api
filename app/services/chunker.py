from dataclasses import dataclass

from app.services.parsers import Section


@dataclass
class Chunk:
    text: str
    page: int | None
    heading: str | None
    chunk_index: int


def chunk_sections(sections: list[Section], chunk_size_words: int, overlap_words: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = max(chunk_size_words - overlap_words, 1)
    chunk_index = 0

    for section in sections:
        words = section.text.split()
        if not words:
            continue

        if len(words) <= chunk_size_words:
            chunks.append(
                Chunk(
                    text=" ".join(words),
                    page=section.page,
                    heading=section.heading,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
            continue

        start = 0
        while start < len(words):
            window = words[start : start + chunk_size_words]
            chunks.append(
                Chunk(
                    text=" ".join(window),
                    page=section.page,
                    heading=section.heading,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
            if start + chunk_size_words >= len(words):
                break
            start += step

    return chunks
