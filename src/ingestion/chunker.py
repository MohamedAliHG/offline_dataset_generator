"""Chunker factory for Docling output."""

from typing import Optional

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer
from docling_core.types.doc import ImageRefMode
from transformers import AutoTokenizer

from src.config import ConfigLoader, config as default_config


def get_chunker(
    tokenizer_model_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    image_mode: Optional[ImageRefMode] = None,
    image_placeholder: Optional[str] = None,
    mark_annotations: Optional[bool] = None,
    include_annotations: Optional[bool] = None,
    cfg: Optional[ConfigLoader] = None,
) -> HybridChunker:
    """Create and configure a ``HybridChunker`` with optional overrides."""
    config_to_use = cfg or default_config

    default_tokenizer = config_to_use.get("model", "tokenizer", default="BAAI/bge-small-en-v1.5")
    default_max_tokens = config_to_use.get("document", "max_tokens", default=512)

    model_id = tokenizer_model_id or default_tokenizer
    tokens = max_tokens or default_max_tokens
    img_mode = image_mode if image_mode is not None else ImageRefMode.PLACEHOLDER
    img_placeholder = image_placeholder if image_placeholder is not None else ""
    mark_annot = mark_annotations if mark_annotations is not None else True
    include_annot = include_annotations if include_annotations is not None else True

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(model_id),
        max_tokens=tokens,
    )

    class CustomMDSerializerProvider(ChunkingSerializerProvider):
        def get_serializer(self, doc):
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=MarkdownTableSerializer(),
                params=MarkdownParams(
                    image_mode=img_mode,
                    image_placeholder=img_placeholder,
                    mark_annotations=mark_annot,
                    include_annotations=include_annot,
                ),
            )

    return HybridChunker(
        tokenizer=tokenizer,
        serializer_provider=CustomMDSerializerProvider(),
    )
