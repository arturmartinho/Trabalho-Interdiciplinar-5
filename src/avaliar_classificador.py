"""Mede a acurácia do classificador sobre um conjunto de prompts rotulados.

Uso:
    python src/avaliar_classificador.py [caminho.csv]

O arquivo (CSV ou JSON) precisa dos campos `id`, `prompt` e `classe_esperada`. Sem argumento,
usa data/prompts_rotulados.csv. Imprime a acurácia, precisão e revocação por
classe, a matriz de confusão e os prompts classificados errado.
"""

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from classificador import CLASSE_CODIGO, CLASSE_COMPLEXO, CLASSE_SIMPLES, classificar

CLASSES = [CLASSE_SIMPLES, CLASSE_CODIGO, CLASSE_COMPLEXO]
CSV_PADRAO = Path(__file__).resolve().parent.parent / "data" / "prompts_rotulados.csv"


@dataclass(frozen=True)
class Avaliacao:
    """Resultado da avaliação sobre o conjunto rotulado."""

    total: int
    acertos: int
    matriz: Dict[str, Dict[str, int]]   # matriz[esperada][prevista]
    erros: List[Dict[str, str]]

    @property
    def acuracia(self) -> float:
        return self.acertos / self.total if self.total else 0.0

    def precisao(self, classe: str) -> float:
        previstas = sum(self.matriz[esp][classe] for esp in CLASSES)
        return self.matriz[classe][classe] / previstas if previstas else 0.0

    def revocacao(self, classe: str) -> float:
        esperadas = sum(self.matriz[classe].values())
        return self.matriz[classe][classe] / esperadas if esperadas else 0.0


def carregar(caminho: Path) -> List[Dict[str, str]]:
    """Lê o CSV, ou o JSON no formato de src/resources/prompts.json."""
    caminho = Path(caminho)
    with open(caminho, encoding="utf-8", newline="") as f:
        if caminho.suffix.lower() == ".json":
            return json.load(f)
        return list(csv.DictReader(f))


def avaliar(linhas: List[Dict[str, str]]) -> Avaliacao:
    matriz = {esp: {prev: 0 for prev in CLASSES} for esp in CLASSES}
    erros = []
    for linha in linhas:
        esperada = linha["classe_esperada"]
        r = classificar(linha["prompt"])
        matriz[esperada][r.classe] += 1
        if r.classe != esperada:
            erros.append({
                "id": linha["id"],
                "prompt": linha["prompt"],
                "esperada": esperada,
                "prevista": r.classe,
                "motivo": r.motivo,
            })
    acertos = sum(matriz[c][c] for c in CLASSES)
    return Avaliacao(len(linhas), acertos, matriz, erros)


def imprimir(av: Avaliacao) -> None:
    print(f"Acuracia: {av.acertos}/{av.total} = {av.acuracia:.1%}\n")

    print(f"{'classe':<10} {'precisao':>9} {'revocacao':>10}")
    for c in CLASSES:
        print(f"{c:<10} {av.precisao(c):>9.1%} {av.revocacao(c):>10.1%}")

    print("\nMatriz de confusao (linhas = esperada, colunas = prevista)")
    print(f"{'':<10}" + "".join(f"{c:>10}" for c in CLASSES))
    for esp in CLASSES:
        print(f"{esp:<10}" + "".join(f"{av.matriz[esp][prev]:>10}" for prev in CLASSES))

    if av.erros:
        print(f"\nErros ({len(av.erros)}):")
        for e in av.erros:
            print(f"  [{e['id']}] {e['esperada']} -> {e['prevista']}: {e['prompt']}")
            print(f"        motivo: {e['motivo']}")


def main() -> None:
    caminho = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV_PADRAO
    imprimir(avaliar(carregar(caminho)))


if __name__ == "__main__":
    main()
