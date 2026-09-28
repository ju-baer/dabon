"""
src/utils/embeddings.py
-----------------------
Embedding utilities for computing geometric diversity metrics.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np


class SBERTEmbedder:
    """SBERT sentence embedder using all-MiniLM-L6-v2 (d=384)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError("Run: pip install sentence-transformers")

    def embed(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Args:
            texts: list of N strings
        Returns:
            embeddings: (N, d) float32 numpy array, L2-normalised
        """
        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)


class LastLayerEmbedder:
    """
    Use the LLM's last hidden state (mean pool over tokens) as embeddings.
    More semantically rich than SBERT for within-model diversity.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16, device_map=self.device
            )
            self._model.eval()

    def embed(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        import torch
        self._load()
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                out = self._model(**enc, output_hidden_states=True)
                # Mean pool last hidden state
                hidden = out.last_hidden_state  # (B, seq, d)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                emb = pooled.float().cpu().numpy()
            # L2 normalise
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            all_embeddings.append(emb / norms)
        return np.vstack(all_embeddings).astype(np.float32)
