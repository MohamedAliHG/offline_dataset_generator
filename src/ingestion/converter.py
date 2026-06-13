"""Document converter factory."""

from typing import Optional

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.config import ConfigLoader, config as default_config


def get_document_converter(
    pdf_pipeline_options=None,
    cfg: Optional[ConfigLoader] = None,
) -> DocumentConverter:
    """Create a ``DocumentConverter`` configured for PDF ingestion."""
    if pdf_pipeline_options is None:
        cfg_to_use = cfg or default_config
        pdf_pipeline_options = cfg_to_use.get_pdf_pipeline_options()

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
        }
    )
