"""Treina o classificador ML (Regressão Logística) sobre o conjunto gerado.

- Treina só com o split "treino" de prompts_gerados.json. O split "teste" e os 40 prompts
  manuais (prompts.json) ficam de fora e servem para a avaliação.
- Escolhe o C da regressão por validação cruzada estratificada (5 dobras) no treino.
- Salva o pipeline + metadados em modelos/classificador_<features>.joblib, que é o
  caminho padrão lido por src/classificador.py (CLASSIFICADOR_MODELO no .env).

Uso:
  python -m src train-classifier                          # TF-IDF (padrão)
  python -m src train-classifier --features embed         # embeddings do Ollama
  python -m src train-classifier --features embed --embed-modelo nomic-embed-text
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAIZ = BASE.parent
sys.path.insert(0, str(RAIZ))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(RAIZ / ".env")

import numpy as np  # noqa: E402

from src.classificador_ml import EMBED_MODELO_PADRAO, EmbeddingsOllama, ModeloML, construir_pipeline  # noqa: E402
from src.texto import tokenizador_ativo  # noqa: E402

GRADE_C = [0.5, 1, 2, 4, 8, 16, 32]


def carregar(caminho: Path):
    return json.loads(caminho.read_text(encoding="utf-8"))


def cache_embed(modelo: str) -> Path:
    return BASE / "dados" / f"cache_embed_{modelo.replace('/', '_').replace(':', '_')}.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", default=str(BASE / "prompts_gerados.json"))
    ap.add_argument("--golden", default=str(BASE / "prompts.json"))
    ap.add_argument("--features", choices=("tfidf", "embed"), default="tfidf")
    ap.add_argument("--embed-modelo", default=EMBED_MODELO_PADRAO)
    ap.add_argument("--C", type=float, default=None, help="fixa o C (sem busca por validação cruzada)")
    ap.add_argument("--saida", default=None, help="padrão: modelos/classificador_<features>.joblib")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import sklearn

    dados = carregar(Path(a.dados))
    treino = [d for d in dados if d.get("split") == "treino"]
    teste = [d for d in dados if d.get("split") == "teste"]
    golden = carregar(Path(a.golden))
    if not treino:
        sys.exit("ERRO: nenhum prompt com split=treino. Gere os dados: python -m src generate-prompts")

    X, y = [d["prompt"] for d in treino], [d["classe_esperada"] for d in treino]
    print(f"Treino: {len(X)} prompts {dict(Counter(y))} | teste: {len(teste)} | manuais: {len(golden)}")
    print(f"Features: {a.features}" + (f" ({a.embed_modelo})" if a.features == "embed" else "")
          + f" | tokenizador: {tokenizador_ativo()}")

    emb = EmbeddingsOllama(modelo=a.embed_modelo) if a.features == "embed" else None
    if emb is not None:
        emb.carregar_cache(cache_embed(a.embed_modelo))
        # calcula tudo uma vez; as dobras da validação cruzada reaproveitam o cache
        emb.transform(X + [d["prompt"] for d in teste] + [g["prompt"] for g in golden])
        emb.salvar_cache(cache_embed(a.embed_modelo))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed)
    resultados_cv = {}
    for C in ([a.C] if a.C else GRADE_C):
        pipe = construir_pipeline(a.features, C=C, embed_modelo=a.embed_modelo)
        notas = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
        resultados_cv[C] = (float(notas.mean()), float(notas.std()))
        print(f"  C={C:<5g} F1-macro CV = {notas.mean():.3f} ± {notas.std():.3f}")
    melhor_C = max(resultados_cv, key=lambda c: resultados_cv[c][0])

    pipe = construir_pipeline(a.features, C=melhor_C, embed_modelo=a.embed_modelo)
    pipe.fit(X, y)

    def avaliar(itens):
        if not itens:
            return None
        pred = pipe.predict([d["prompt"] for d in itens])
        verdade = [d["classe_esperada"] for d in itens]
        return {"n": len(itens), "acuracia": round(float(accuracy_score(verdade, pred)), 4),
                "f1_macro": round(float(f1_score(verdade, pred, average="macro")), 4)}

    av_teste, av_golden = avaliar(teste), avaliar(golden)
    print(f"Melhor C={melhor_C} | teste: {av_teste} | manuais: {av_golden}")

    meta = {
        "treinado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": a.features, "embed_modelo": a.embed_modelo if a.features == "embed" else None,
        "C": melhor_C, "n_treino": len(X), "classes": sorted(set(y)),
        "acuracia_validacao": {"cv_f1_macro": resultados_cv[melhor_C][0], "teste": av_teste, "manuais": av_golden},
        "grade_cv": {str(k): v for k, v in resultados_cv.items()},
        "tokenizador": tokenizador_ativo(), "sklearn": sklearn.__version__, "numpy": np.__version__,
        "dados": str(Path(a.dados).name),
    }
    saida = Path(a.saida) if a.saida else RAIZ / "modelos" / f"classificador_{a.features}.joblib"
    ModeloML(pipeline=pipe, features=a.features, metadados=meta).salvar(saida)
    saida.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modelo salvo em {saida} (metadados em {saida.with_suffix('.json').name})")


if __name__ == "__main__":
    main()
