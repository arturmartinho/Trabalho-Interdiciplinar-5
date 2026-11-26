"""Classificador treinado: características do texto + Regressão Logística.

Dois extratores de características:

- ``tfidf`` (padrão): TF-IDF de palavras (texto normalizado, 1-2 gramas) + TF-IDF de
  n-gramas de caracteres (2-5, preserva símbolos de código como ``{``, ``;``, ``()``) +
  log do número de tokens. Roda em ~1 ms, sem rede e sem memória extra no Ollama.
- ``embed``: embeddings de frase do próprio Ollama (``/api/embed``; padrão
  ``paraphrase-multilingual``) + log do número de tokens. Melhor com paráfrases e
  idiomas variados, mas custa uma chamada HTTP por prompt e um modelo a mais em RAM.

O modelo treinado é salvo com ``joblib`` (``experiments/treinar_classificador.py``) junto
com metadados (data, tamanho do treino, acurácia de validação, versão do scikit-learn).
Arquivos ``.joblib`` são pickles: carregue apenas arquivos gerados por você.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .texto import contar_tokens, normalizar

FEATURES = ("tfidf", "embed")
EMBED_MODELO_PADRAO = "paraphrase-multilingual"
_CACHE_EMBED: Dict[str, Dict[str, np.ndarray]] = {}

try:  # scikit-learn é opcional: sem ele, o classificador por regras continua funcionando
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:  # pragma: no cover - coberto pelo fallback em classificador.py
    BaseEstimator = object  # type: ignore[assignment,misc]
    TransformerMixin = object  # type: ignore[assignment,misc]


# --------------------------------------------------------------------------- #
# Transformadores (precisam ficar no nível do módulo para o pickle funcionar)
# --------------------------------------------------------------------------- #
class ComprimentoLog(BaseEstimator, TransformerMixin):
    """Uma coluna: log(1 + tokens) / 6, em escala parecida com a do TF-IDF."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([[math.log1p(contar_tokens(t)) / 6.0] for t in X])


class EmbeddingsOllama(BaseEstimator, TransformerMixin):
    """Embeddings normalizados vindos do endpoint ``/api/embed`` do Ollama."""

    def __init__(self, modelo: str = EMBED_MODELO_PADRAO, host: Optional[str] = None, lote: int = 32):
        self.modelo = modelo
        self.host = host
        self.lote = lote

    def _cache(self) -> Dict[str, np.ndarray]:
        # Cache no nível do módulo (por modelo): os clones que o scikit-learn cria na
        # validação cruzada compartilham os embeddings já calculados.
        return _CACHE_EMBED.setdefault(self.modelo, {})

    @staticmethod
    def _chave(texto: str) -> str:
        return hashlib.sha1(texto.encode("utf-8")).hexdigest()

    def carregar_cache(self, caminho: Path) -> None:
        if caminho.exists():
            d = np.load(caminho)
            self._cache().update({k: d[k] for k in d.files})

    def salvar_cache(self, caminho: Path) -> None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(caminho, **self._cache())

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import httpx

        textos = list(X)
        cache = self._cache()
        faltando = [t for t in dict.fromkeys(textos) if self._chave(t) not in cache]
        host = (self.host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        for i in range(0, len(faltando), self.lote):
            parte = faltando[i: i + self.lote]
            r = httpx.post(f"{host}/api/embed", json={"model": self.modelo, "input": parte}, timeout=120)
            r.raise_for_status()
            for t, v in zip(parte, r.json()["embeddings"]):
                v = np.asarray(v, dtype=np.float32)
                cache[self._chave(t)] = v / (np.linalg.norm(v) or 1.0)
        return np.vstack([cache[self._chave(t)] for t in textos])


# --------------------------------------------------------------------------- #
# Construção, treino e uso
# --------------------------------------------------------------------------- #
def construir_pipeline(features: str = "tfidf", C: float = 4.0, embed_modelo: str = EMBED_MODELO_PADRAO):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    if features == "tfidf":
        # Sem a feature de tamanho: nos dados gerados quase todo prompt complexo é longo, e com ela o
        # modelo aprendia "curto = simples" (acurácia nos 40 manuais: 95% com, 97,5% sem). O limite de
        # tamanho fica com o híbrido (> LIMITE_LONGO tokens nunca vai para o modelo simples).
        partes = [
            ("palavras", TfidfVectorizer(preprocessor=normalizar, ngram_range=(1, 2), sublinear_tf=True)),
            ("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
        ]
    elif features == "embed":
        # Aqui o tamanho ajuda (manuais: 97,5% com, 92,5% sem).
        partes = [("embeddings", EmbeddingsOllama(modelo=embed_modelo)), ("tamanho", ComprimentoLog())]
    else:
        raise ValueError(f"features desconhecidas: {features} (use {FEATURES})")
    return Pipeline([
        ("features", FeatureUnion(partes)),
        ("lr", LogisticRegression(C=C, max_iter=5000, class_weight="balanced")),
    ])


@dataclass
class ModeloML:
    pipeline: object
    features: str
    metadados: dict

    @property
    def classes(self) -> List[str]:
        return list(self.pipeline.classes_)

    def prever(self, texto: str) -> tuple[str, float, Dict[str, float]]:
        probs = self.pipeline.predict_proba([texto])[0]
        i = int(np.argmax(probs))
        return self.classes[i], float(probs[i]), {c: round(float(p), 4) for c, p in zip(self.classes, probs)}

    def prever_lote(self, textos: Iterable[str]) -> tuple[List[str], np.ndarray]:
        probs = self.pipeline.predict_proba(list(textos))
        return [self.classes[i] for i in probs.argmax(axis=1)], probs.max(axis=1)

    def salvar(self, caminho: Path) -> None:
        import joblib

        caminho.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "features": self.features, "metadados": self.metadados}, caminho)


def carregar(caminho: str | Path) -> ModeloML:
    """Carrega um modelo salvo. Use ``carregar_cache`` no caminho de execução."""
    import joblib

    d = joblib.load(caminho)
    return ModeloML(pipeline=d["pipeline"], features=d["features"], metadados=d.get("metadados", {}))


@lru_cache(maxsize=4)
def _carregar_cache(caminho: str, mtime: float) -> ModeloML:
    return carregar(caminho)


def carregar_cache(caminho: str | Path) -> ModeloML:
    """Igual a ``carregar``, mas reaproveita o objeto enquanto o arquivo não mudar."""
    p = Path(caminho)
    return _carregar_cache(str(p.resolve()), p.stat().st_mtime)
