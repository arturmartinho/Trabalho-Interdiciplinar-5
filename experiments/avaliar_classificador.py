"""Compara as estratégias de classificação em acurácia, tipo de erro e tempo.

Estratégias avaliadas:
  regras_palavras : classificador antigo (25/120 palavras), para comparação
  regras          : regras com limites em tokens
  ml_<features>   : só o modelo treinado (para cada modelos/classificador_*.joblib)
  hibrido_<feat.> : sintaxe de código -> regras; resto -> modelo; incerto -> regras
  llm             : (opcional, --llm TAG) um LLM do Ollama classifica zero-shot

Conjuntos:
  teste   : split "teste" de prompts_gerados.json (mesma distribuição do treino)
  manuais : os 40 prompts escritos à mão (prompts.json), fora da distribuição do gerador

Tipos de erro: "subdimensionado" = foi para um modelo menor que o necessário
(complexo->codigo/simples, codigo->simples), que arrisca a qualidade da resposta;
"superdimensionado" = foi para um modelo maior, que só custa latência.

Saídas: relatorio_classificador.md, avaliacao_classificador.json e
graficos/grafico_classificadores.png.

Uso:
  python -m src eval-classifier
  python -m src eval-classifier --llm qwen2.5:0.5b --llm-max 150
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAIZ = BASE.parent
sys.path.insert(0, str(RAIZ))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(RAIZ / ".env")

import numpy as np  # noqa: E402

from src import classificador as cl  # noqa: E402
from src.texto import tokenizador_ativo  # noqa: E402

CLASSES = list(cl.CLASSES)
POSTO = {"simples": 0, "codigo": 1, "complexo": 2}  # porte do modelo de destino

PROMPT_LLM = """Classifique o pedido do usuário em exatamente uma classe:
- "simples": pergunta curta e factual ou tarefa trivial com resposta curta.
- "codigo": qualquer pedido de programação (escrever, corrigir, explicar ou testar código, SQL, regex, terminal).
- "complexo": exige raciocínio em várias etapas, análise, argumentação, prova ou texto longo.

Pedido:
\"\"\"{prompt}\"\"\"

Responda só com JSON: {{"classe": "simples" | "codigo" | "complexo"}}"""


def classificador_llm(modelo: str, host: str):
    import httpx

    client = httpx.Client(timeout=300)
    schema = {"type": "object", "properties": {"classe": {"type": "string", "enum": CLASSES}}, "required": ["classe"]}

    def f(prompt: str) -> str:
        r = client.post(f"{host}/api/generate", json={
            "model": modelo, "prompt": PROMPT_LLM.format(prompt=prompt[:4000]), "stream": False,
            "format": schema, "think": False, "options": {"temperature": 0, "num_predict": 20}})
        r.raise_for_status()
        try:
            c = json.loads(r.json().get("response") or "{}").get("classe")
        except json.JSONDecodeError:
            c = None
        return c if c in CLASSES else "complexo"  # resposta inválida -> padrão conservador
    return f


def estrategias(args):
    est = {"regras_palavras": lambda p: cl.classificar_regras(p, unidade="palavras")[0],
           "regras": lambda p: cl.classificar(p, estrategia="regras").classe}
    for arq in sorted((RAIZ / "modelos").glob("classificador_*.joblib")):
        feat = arq.stem.replace("classificador_", "")
        est[f"ml_{feat}"] = (lambda caminho: lambda p: cl.classificar(p, estrategia="ml", caminho_modelo=caminho).classe)(str(arq))
        est[f"hibrido_{feat}"] = (lambda caminho: lambda p: cl.classificar(p, estrategia="hibrido", caminho_modelo=caminho).classe)(str(arq))
    if args.llm:
        est[f"llm_{args.llm}"] = classificador_llm(args.llm, args.ollama)
    return est


def metricas(itens, pred):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    v = [i["classe_esperada"] for i in itens]
    p, r, f, _ = precision_recall_fscore_support(v, pred, labels=CLASSES, zero_division=0)
    sub = sum(1 for a, b in zip(v, pred) if POSTO[b] < POSTO[a])
    sup = sum(1 for a, b in zip(v, pred) if POSTO[b] > POSTO[a])
    grupos = {}
    for campo in ("idioma", "subtipo", "adversarial", "linguagem"):
        acc = defaultdict(lambda: [0, 0])
        for it, pr in zip(itens, pred):
            k = str(it.get(campo, ""))
            if campo == "linguagem" and not k:
                continue
            acc[k][0] += it["classe_esperada"] == pr
            acc[k][1] += 1
        grupos[campo] = {k: round(c / n, 4) for k, (c, n) in sorted(acc.items())}
    return {
        "n": len(itens), "acuracia": round(float(accuracy_score(v, pred)), 4),
        "f1_macro": round(float(f1_score(v, pred, labels=CLASSES, average="macro", zero_division=0)), 4),
        "precisao": dict(zip(CLASSES, map(lambda x: round(float(x), 4), p))),
        "revocacao": dict(zip(CLASSES, map(lambda x: round(float(x), 4), r))),
        "f1": dict(zip(CLASSES, map(lambda x: round(float(x), 4), f))),
        "subdimensionado": round(sub / len(itens), 4), "superdimensionado": round(sup / len(itens), 4),
        "confusao": confusion_matrix(v, pred, labels=CLASSES).tolist(),
        "por": grupos,
    }


def rodar(nome, f, itens):
    pred, tempos = [], []
    for it in itens:
        t0 = time.perf_counter()
        pred.append(f(it["prompt"]))
        tempos.append((time.perf_counter() - t0) * 1000)
    m = metricas(itens, pred)
    m["tempo_ms"] = {"media": round(float(np.mean(tempos)), 3), "p95": round(float(np.percentile(tempos, 95)), 3)}
    erros = [(it["id"], it["classe_esperada"], pr, it["prompt"]) for it, pr in zip(itens, pred) if pr != it["classe_esperada"]]
    return m, erros


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    import os

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", default=str(BASE / "prompts_gerados.json"))
    ap.add_argument("--golden", default=str(BASE / "prompts.json"))
    ap.add_argument("--llm", default="", help="tag do Ollama para avaliar um LLM como classificador")
    ap.add_argument("--llm-max", type=int, default=150, help="máximo de prompts de teste para o LLM (é lento)")
    ap.add_argument("--ollama", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"))
    ap.add_argument("--relatorio", default=str(BASE / "relatorio_classificador.md"))
    ap.add_argument("--exemplos-erro", type=int, default=12)
    a = ap.parse_args()

    dados = json.loads(Path(a.dados).read_text(encoding="utf-8"))
    conjuntos = {"teste": [d for d in dados if d.get("split") == "teste"],
                 "manuais": json.loads(Path(a.golden).read_text(encoding="utf-8"))}
    if not conjuntos["teste"]:
        sys.exit("ERRO: nenhum prompt com split=teste. Gere os dados: python -m src generate-prompts")

    res, erros = defaultdict(dict), defaultdict(dict)
    from src import classificador_ml

    for nome, f in estrategias(a).items():
        # O cache de embeddings é do módulo: sem limpar, hibrido_embed reaproveitaria os vetores
        # calculados por ml_embed e o tempo medido sairia ~1 ms em vez de ~450 ms.
        classificador_ml._CACHE_EMBED.clear()
        for cj, itens in conjuntos.items():
            if nome.startswith("llm_") and cj == "teste" and len(itens) > a.llm_max:
                itens = sorted(itens, key=lambda i: i["id"])[:: max(1, len(itens) // a.llm_max)][: a.llm_max]
            res[nome][cj], erros[nome][cj] = rodar(nome, f, itens)
            m = res[nome][cj]
            print(f"{nome:18s} {cj:8s} n={m['n']:4d} acc={pct(m['acuracia']):>6s} F1={m['f1_macro']:.3f} "
                  f"sub={pct(m['subdimensionado']):>6s} sup={pct(m['superdimensionado']):>6s} "
                  f"t={m['tempo_ms']['media']:.2f}ms")

    saida_json = Path(a.relatorio).with_name("avaliacao_classificador.json")
    saida_json.write_text(json.dumps({"tokenizador": tokenizador_ativo(), "limiar_confianca": cl.LIMIAR_CONFIANCA,
                                      "resultados": res}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- gráfico ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nomes = list(res)
        x = np.arange(len(nomes))
        fig, ax = plt.subplots(figsize=(max(6.5, 1.1 * len(nomes)), 4))
        for i, (cj, cor) in enumerate((("teste", "#1f77b4"), ("manuais", "#8c8c8c"))):
            b = ax.bar(x + (i - 0.5) * 0.38, [res[n][cj]["acuracia"] * 100 for n in nomes], 0.38,
                       label=f"{cj} (n={res[nomes[0]][cj]['n']})", color=cor)
            ax.bar_label(b, fmt="%.0f", fontsize=8)
        ax.set_xticks(x, nomes, rotation=20, ha="right")
        ax.set_ylabel("Acurácia (%)")
        ax.set_ylim(0, 105)
        ax.set_title("Acurácia do classificador por estratégia", pad=18)
        ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
        fig.tight_layout()
        (BASE / "graficos").mkdir(exist_ok=True)
        fig.savefig(BASE / "graficos" / "grafico_classificadores.png", dpi=200)
        plt.close(fig)
    except ImportError:
        pass

    # ---------- relatório ----------
    L = ["# Avaliação do classificador de prompts\n",
         "> Gerado por `experiments/avaliar_classificador.py`.\n",
         f"- Tokenizador: `{tokenizador_ativo()}` | limites: {cl.LIMITE_CURTO}/{cl.LIMITE_LONGO} tokens | "
         f"limiar de confiança do híbrido: {cl.LIMIAR_CONFIANCA}",
         f"- Teste: {res['regras']['teste']['n']} prompts gerados (split de teste) | manuais: "
         f"{res['regras']['manuais']['n']} prompts escritos à mão",
         "- *Subdimensionado* = enviado a um modelo menor que o necessário (risco de qualidade); "
         "*superdimensionado* = a um maior (custa latência).\n"]
    for cj in ("teste", "manuais"):
        L.append(f"## Conjunto: {cj}\n")
        L.append("| Estratégia | n | Acurácia | F1-macro | Subdim. | Superdim. | Tempo médio (ms) | P95 (ms) |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for n in res:
            m = res[n][cj]
            L.append(f"| {n} | {m['n']} | {pct(m['acuracia'])} | {m['f1_macro']:.3f} | {pct(m['subdimensionado'])} | "
                     f"{pct(m['superdimensionado'])} | {m['tempo_ms']['media']:.2f} | {m['tempo_ms']['p95']:.2f} |")
        L.append("")
    melhor = max(res, key=lambda n: res[n]["teste"]["f1_macro"])
    m = res[melhor]["teste"]
    L.append(f"## Detalhe da melhor estratégia no teste: `{melhor}`\n")
    L.append("Matriz de confusão (linhas = esperado, colunas = previsto):\n")
    L.append("| | " + " | ".join(CLASSES) + " |\n|---|" + "---:|" * len(CLASSES))
    for c, linha in zip(CLASSES, m["confusao"]):
        L.append(f"| **{c}** | " + " | ".join(map(str, linha)) + " |")
    for campo, titulo in (("idioma", "idioma"), ("subtipo", "subtipo"), ("adversarial", "caso difícil (adversarial)"),
                          ("linguagem", "linguagem de programação")):
        L.append(f"\nAcurácia por {titulo}, regras x `{melhor}`:\n")
        L.append(f"| {titulo} | regras | {melhor} |\n|---|---:|---:|")
        for k, v in m["por"][campo].items():
            L.append(f"| {k} | {pct(res['regras']['teste']['por'][campo].get(k, 0))} | {pct(v)} |")
    L.append(f"\n## Exemplos de erro de `{melhor}` (teste)\n")
    for pid, esp, pr, txt in erros[melhor]["teste"][: a.exemplos_erro]:
        L.append(f"- `{pid}` esperado **{esp}**, previsto **{pr}**: {txt[:180].replace(chr(10), ' ')}")
    L.append("\n## Limitações\n")
    L.append("- O rótulo dos prompts gerados é a classe pedida ao LLM gerador, não uma anotação humana; "
             "revise uma amostra antes de usar os números no artigo.")
    L.append("- Treino e teste vêm do mesmo gerador e do mesmo template: a acurácia no teste tende a ser otimista. "
             "O conjunto manual é a medida fora da distribuição.")
    Path(a.relatorio).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Relatório em {a.relatorio} | JSON em {saida_json.name} | gráfico em graficos/grafico_classificadores.png")


if __name__ == "__main__":
    main()
