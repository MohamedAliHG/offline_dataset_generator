"""Configuration loader for the dataset pipeline."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from dotenv import load_dotenv
from src.ingestion.prompts.default_vlm_prompts import DEFAULT_VLM_PROMPT


class ConfigLoader:
    """Configuration loader for the C-130 dataset pipeline."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        allow_new_keys: bool = True,
    ) -> None:
        self._config: Dict[str, Any] = {}
        self._allow_new_keys = allow_new_keys
        self._load_config(config_path)

    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Load configuration from YAML and environment variables."""
        root_dir = Path(__file__).resolve().parent.parent
        env_path = root_dir / ".env"
        load_dotenv(env_path)

        yaml_path = Path(config_path) if config_path else (root_dir / "config" / "config.yaml")
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

    def reload(self, config_path: Optional[Union[str, Path]] = None) -> None:
        self._load_config(config_path)

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def get(self, *keys: str, default: Any = None) -> Any:
        value: Any = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, *keys_and_value: Any, force: bool = False) -> None:
        if len(keys_and_value) < 2:
            raise ValueError("At least one key and a value must be provided")

        keys = keys_and_value[:-1]
        value = keys_and_value[-1]

        if not self._allow_new_keys and not force:
            self._validate_key_exists(keys)

        current: Dict[str, Any] = self._config
        for key in keys[:-1]:
            if key not in current:
                if not self._allow_new_keys and not force:
                    full_path = ".".join(str(k) for k in keys)
                    raise KeyError(
                        f"Key path '{full_path}' does not exist and allow_new_keys is False. "
                        "Use force=True to override."
                    )
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        if keys[-1] not in current and not self._allow_new_keys and not force:
            prefix = ".".join(str(k) for k in keys[:-1])
            raise KeyError(
                f"Key '{keys[-1]}' does not exist in path '{prefix}' and allow_new_keys is False."
            )

        current[keys[-1]] = value

    def _validate_key_exists(self, keys: tuple[Any, ...]) -> None:
        current: Any = self._config
        for i, key in enumerate(keys):
            if not isinstance(current, dict) or key not in current:
                sub_path = ".".join(str(k) for k in keys[: i + 1])
                raise KeyError(
                    f"Key path '{sub_path}' does not exist and allow_new_keys is False. "
                    "Use force=True to override."
                )
            current = current[key]

    def set_allow_new_keys(self, allow: bool) -> None:
        self._allow_new_keys = allow

    def get_allow_new_keys(self) -> bool:
        return self._allow_new_keys

    def resolve_doc_ref(self, doc_id: str, override: Optional[str] = None) -> str:
        """Resolve a human-readable document reference for citations."""
        if override:
            return override

        mapped = self.get("dataset_gen", "doc_ref_map", default={})
        if isinstance(mapped, dict):
            value = mapped.get(doc_id)
            if isinstance(value, str) and value.strip():
                return value.strip()

        fallback = self.get("dataset_gen", "doc_ref", default=None)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()

        return doc_id

    def get_pdf_pipeline_options(self):
        """Build Docling ``PdfPipelineOptions`` using configured VLM settings."""
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            PictureDescriptionApiOptions,
        )

        vlm_url = self.get("vlm", "url")
        if not vlm_url:
            raise ValueError(
                "[vlm] url is required in config/config.yaml. Example: 'http://localhost:8080'"
            )

        model_name = self.get("vlm", "model_name", default="llava")
        timeout = self.get("vlm", "timeout", default=60)
        prompt = DEFAULT_VLM_PROMPT or self.get(
            "vlm",
            "prompt",
            default="Describe this image in sentences in a single paragraph.",
        )
        image_scale = self.get("document", "image_resolution_scale", default=2)

        picture_desc_options = PictureDescriptionApiOptions(
            url=f"{vlm_url.rstrip('/')}/v1/chat/completions",
            prompt=prompt,
            params={"model": model_name},
            headers={"Authorization": "Bearer not-needed"},
            timeout=timeout,
        )

        return PdfPipelineOptions(
            images_scale=image_scale,
            generate_picture_images=True,
            do_picture_description=True,
            picture_description_options=picture_desc_options,
            enable_remote_services=True,
        )


config = ConfigLoader()
