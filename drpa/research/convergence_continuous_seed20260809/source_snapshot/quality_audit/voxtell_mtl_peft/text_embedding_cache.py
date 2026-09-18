"""Offline prompt embedding cache using the official VoxTell predictor path."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

from voxtell.inference.predictor import VoxTellPredictor


class TextEmbeddingCache:
    def __init__(self, bank_path: str, model_dir: str, cache_path: str | None = None) -> None:
        self.bank_path = str(bank_path)
        self.model_dir = str(model_dir)
        self.cache_path = str(cache_path) if cache_path else None
        self._values = {}
        self._load_npz(self.bank_path)
        if self.cache_path and Path(self.cache_path).is_file():
            self._load_npz(self.cache_path)

    def _load_npz(self, path: str) -> None:
        data = np.load(path)
        labels = data["labels"].tolist()
        values = data["embeddings"].astype(np.float32)
        for label, value in zip(labels, values):
            self._values[str(label).lower()] = value

    def get(self, prompts: Iterable[str], device: torch.device) -> torch.Tensor:
        prompts = [str(x).lower() for x in prompts]
        missing = [p for p in prompts if p not in self._values]
        if missing:
            # This follows the official predictor's Qwen path. The predictor's
            # embedding functions are inference_mode-only, so Qwen is never
            # part of the backward graph.
            predictor = VoxTellPredictor(
                model_dir=self.model_dir,
                device=torch.device("cpu"),
                embedding_bank=self.bank_path,
                use_precomputed_embeddings=True,
            )
            computed = predictor.embed_text_prompts(missing)[0].cpu().numpy().astype(np.float32)
            for prompt, value in zip(missing, computed):
                self._values[prompt] = value
            if self.cache_path:
                self.save(self.cache_path)
        values = np.stack([self._values[p] for p in prompts], axis=0)
        out = torch.from_numpy(values).to(device=device, dtype=torch.float32).unsqueeze(0)
        out.requires_grad_(False)
        return out

    def save(self, path: str) -> None:
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        labels = sorted(self._values)
        values = np.stack([self._values[x] for x in labels]).astype(np.float16)
        np.savez_compressed(path, labels=np.asarray(labels), embeddings=values)

    def contains(self, prompt: str) -> bool:
        return str(prompt).lower() in self._values

    def dimension(self) -> int:
        if not self._values:
            raise RuntimeError("Embedding cache is empty")
        return int(next(iter(self._values.values())).shape[-1])
