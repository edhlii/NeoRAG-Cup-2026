"""Làm sạch nhẹ, chia theo tokenizer và bảo toàn ID tài liệu gốc."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class Chunk:
    id: str
    parent_id: int
    text: str
    club: str
    topic: str


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()


def ingest(corpus, tokenizer, chunk_size=300, chunk_overlap=40) -> list[Chunk]:
    if not corpus or len({d.id for d in corpus}) != len(corpus):
        raise ValueError("Corpus phải có tài liệu và các ID không trùng nhau.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda s: len(tokenizer.encode(s, add_special_tokens=False)),
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
        keep_separator=True,
    )
    chunks = []
    for doc in corpus:
        source = LCDocument(
            page_content=clean_text(doc.text),
            metadata={
                "parent_id": doc.id,
                "club": doc.club,
                "topic": doc.topic,
            },
        )
        if not source.page_content:
            raise ValueError(f"Tài liệu {doc.id} rỗng sau khi làm sạch.")
        for part in splitter.split_documents([source]):
            text = f"{doc.club} | {doc.topic}\n{part.page_content}"
            # Không cho metadata làm đoạn vượt cửa sổ embedding.
            if len(tokenizer.encode("passage: " + text, add_special_tokens=True)) > 512:
                raise ValueError(f"Chunk của tài liệu {doc.id} vượt 512 token; giảm chunk_size.")
            chunks.append(
                Chunk(
                    str(uuid5(NAMESPACE_URL, f"neorag:{doc.id}:{text}")),
                    doc.id,
                    text,
                    doc.club,
                    doc.topic,
                )
            )
    return list({c.id: c for c in chunks}.values())
