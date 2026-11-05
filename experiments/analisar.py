"""Lê resultados.csv (+ metadados.json), calcula métricas, gera 2 gráficos e relatorio_resultados.md.

Métricas descritivas apenas (média, mediana, P95, P99, máximo). A análise estatística
completa (testes, intervalos de confiança, outliers) fica para a semana 5.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CENARIOS = ["baseline", "proposta"]
CLASSES = ["simples", "codigo", "complexo"]
NOME_CLASSE = {"simples": "Simples", "codigo": "Código", "complexo": "Complexo"}


def carregar(caminho):
    with open(caminho, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metricas(vals):
    v = np.array(vals, dtype=float)
    return {"n": len(v), "media": v.mean(), "mediana": float(np.median(v)), "dp": v.std(ddof=1) if len(v) > 1 else 0.0,
            "p95": np.percentile(v, 95), "p99": np.percentile(v, 99), "max": v.max()}


def br(x, casas=2):
    return f"{x:.{casas}f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent
    ap.add_argument("--entrada", default=str(base / "resultados.csv"))
    ap.add_argument("--pasta", default=str(base / "graficos"))
    ap.add_argument("--relatorio", default=None)
    a = ap.parse_args()
    entrada = Path(a.entrada)
    pasta = Path(a.pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    rel_path = Path(a.relatorio) if a.relatorio else entrada.with_name("relatorio_resultados.md")
    meta_path = entrada.with_name(entrada.stem + ".metadados.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    linhas = carregar(entrada)
    total, falhas = defaultdict(int), defaultdict(int)
    lat, lat_cl, chars, chars_cl = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    por_prompt = defaultdict(lambda: defaultdict(list))
    por_passo = defaultdict(list)
    por_passo_cl = defaultdict(list)
    erros = Counter()
    for r in linhas:
        c = r["cenario"]
        total[c] += 1
        if r["sucesso"] != "True":
            falhas[c] += 1
            erros[(c, r["erro"].split(":")[0])] += 1
            continue
        x = float(r["latencia_s"])
        lat[c].append(x)
        lat_cl[(c, r["classe_esperada"])].append(x)
        if r.get("passo"):
            por_passo[(c, r["passo"])].append(x)
            por_passo_cl[(c, r["passo"], r["classe_esperada"])].append(x)
        por_prompt[r["prompt_id"]][c].append(x)
        if r["resposta_chars"]:
            chars[c].append(int(r["resposta_chars"]))
            chars_cl[(c, r["classe_esperada"])].append(int(r["resposta_chars"]))

    vazios = [c for c in CENARIOS if not lat[c]]
    if vazios:
        print(f"Sem nenhuma requisição bem-sucedida em: {', '.join(vazios)}. Total/falhas: {dict(total)} / {dict(falhas)}.")
        print("Primeiros erros:", *[r["erro"] for r in linhas if r["sucesso"] != "True"][:3], sep="\n  ")
        raise SystemExit(1)

    res = {c: metricas(lat[c]) for c in CENARIOS}
    prop = [r for r in linhas if r["cenario"] == "proposta" and r["sucesso"] == "True"]
    desvios = sum(1 for r in prop if r["desvio"] == "True")
    certos = sum(1 for r in prop if r["classe"] == r["classe_esperada"])
    atendido = Counter(r["modelo_atendeu"] for r in prop)
    confusao = Counter((r["classe_esperada"], r["classe"]) for r in prop)
    erradas = sorted({(r["prompt_id"], r["classe_esperada"], r["classe"]) for r in prop if r["classe"] != r["classe_esperada"]})
    overhead = [float(r["latencia_s"]) - float(r["lb_e2e_s"]) for r in prop if r["lb_e2e_s"] not in ("", None)]

    # comparação pareada por prompt (mesmo prompt nos dois cenários)
    difs = [np.mean(d["baseline"]) - np.mean(d["proposta"]) for d in por_prompt.values() if d["baseline"] and d["proposta"]]
    mais_rapidos = sum(1 for d in difs if d > 0)
    red = 1 - res["proposta"]["media"] / res["baseline"]["media"]

    # ---------- console ----------
    print(f"{'cenario':10s} {'n_ok':>5s} {'falhas':>6s} {'media':>8s} {'mediana':>8s} {'P95':>8s} {'P99':>8s} {'max':>8s}")
    for c in CENARIOS:
        m = res[c]
        print(f"{c:10s} {m['n']:5d} {falhas[c]:6d} {m['media']:8.2f} {m['mediana']:8.2f} {m['p95']:8.2f} {m['p99']:8.2f} {m['max']:8.2f}")
    print(f"Redução da latência média (proposta vs baseline): {red:.1%}")
    print(f"Desvios para o fallback na proposta: {desvios} de {len(prop)}")
    print(f"Classe atribuída = classe esperada: {certos} de {len(prop)} ({certos / len(prop):.1%})")
    for k, v in sorted(atendido.items()):
        print(f"  atendido por {k}: {v}")

    # ---------- avisos automáticos ----------
    avisos = []
    cfg = meta.get("config", {})
    if meta and not cfg.get("num_predict"):
        avisos.append("`OLLAMA_NUM_PREDICT` não estava definido: a latência mistura velocidade do modelo com o tamanho da resposta gerada.")
    if not meta:
        avisos.append("Sem `metadados.json`: não dá para confirmar limite de tokens, aquecimento, ambiente nem modelos carregados.")
    if sum(falhas.values()):
        avisos.append(f"Houve falhas ({dict(falhas)}); elas ficaram fora das latências. Tipos: {dict(erros)}.")
    if desvios:
        avisos.append(f"{desvios} requisição(ões) da proposta foram redirecionadas ao fallback; a latência delas inclui duas tentativas.")
    n_min = min(res[c]["n"] for c in CENARIOS)
    if n_min < 100:
        avisos.append(f"Com n={n_min} por cenário, o P99 é praticamente o valor máximo e o P95 depende de poucas observações; use apenas como descrição.")
    if res["proposta"]["media"] >= res["baseline"]["media"]:
        avisos.append("A proposta **não** foi mais rápida na média. Verifique desvios, carga/descarga de modelos no Ollama e o tamanho das respostas antes de interpretar.")
    for cl in CLASSES:
        cb, cp = chars_cl.get(("baseline", cl)), chars_cl.get(("proposta", cl))
        if cb and cp and np.mean(cp) < 0.5 * np.mean(cb):
            avisos.append(f"Classe {cl}: respostas da proposta têm em média menos da metade dos caracteres do baseline "
                          f"({np.mean(cp):.0f} vs {np.mean(cb):.0f}); parte da diferença de latência pode vir do tamanho da resposta, não do roteamento.")
    deriva = {}
    for c in CENARIOS:
        p1, p2 = por_passo.get((c, "1")), por_passo.get((c, "2"))
        if p1 and p2:
            razoes = [np.mean(por_passo_cl[(c, "2", k)]) / np.mean(por_passo_cl[(c, "1", k)])
                      for k in CLASSES if por_passo_cl.get((c, "1", k)) and por_passo_cl.get((c, "2", k))]
            ajust = float(np.mean(razoes)) - 1 if razoes else 0.0   # compara dentro de cada classe
            deriva[c] = (np.mean(p1), np.mean(p2), len(p1), len(p2), ajust)
            if abs(ajust) > 0.15:
                avisos.append(f"Deriva entre passos no cenário {c}: {ajust:+.0%} (média das variações dentro de cada classe, passo 2 vs passo 1). "
                              "Pode ser aquecimento térmico, carga na máquina ou recarga de modelos; como os dois cenários se alternam "
                              "nos passos, o efeito atinge ambos, mas vale investigar.")
    carregados = [m["nome"] for m in meta.get("modelos_carregados_apos_aquecimento", [])]
    if meta and len(carregados) < 3:
        avisos.append(f"Após o aquecimento, só {len(carregados)} modelo(s) estavam carregados ({carregados}); o Ollama pode ter descarregado modelos durante a rodada.")

    # ---------- gráficos ----------
    cores = {"baseline": "#8c8c8c", "proposta": "#1f77b4"}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(3)
    for i, c in enumerate(CENARIOS):
        vals = [res[c]["media"], res[c]["p95"], res[c]["p99"]]
        b = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=c.capitalize(), color=cores[c])
        ax.bar_label(b, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, ["Média", "P95", "P99"])
    ax.set_ylabel("Latência (s)")
    ax.set_title(f"Latência: baseline vs. proposta (n={res['baseline']['n']} ok por cenário)", pad=18)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(pasta / "grafico1_latencia_geral.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, c in enumerate(CENARIOS):
        vals = [np.mean(lat_cl[(c, k)]) if lat_cl[(c, k)] else 0 for k in CLASSES]
        b = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=c.capitalize(), color=cores[c])
        ax.bar_label(b, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, [NOME_CLASSE[k] for k in CLASSES])
    ax.set_ylabel("Latência média (s)")
    ax.set_title("Latência média por classe de prompt", pad=18)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(pasta / "grafico2_latencia_por_classe.png", dpi=200)
    plt.close(fig)
    print(f"Gráficos em {pasta}/")

    # ---------- relatório ----------
    amb = meta.get("ambiente", {})
    arg = meta.get("argumentos", {})
    L = []
    L.append("# Relatório da rodada de experimentos (Aluno 4)\n")
    L.append("> Gerado automaticamente por `analisar.py`. Números descritivos de uma amostra pequena; a análise estatística completa fica para a semana 5.\n")
    L.append("## 1. Configuração e ambiente\n")
    if meta:
        L.append(f"- Início/fim (UTC): {meta.get('inicio_utc')} / {meta.get('fim_utc')}")
        L.append(f"- Repetições: {arg.get('repeticoes')} | prompts: {arg.get('n_prompts')} | semente da ordem: {arg.get('seed')}")
        L.append(f"- Delineamento: {meta.get('delineamento', 'não registrado')}")
        L.append(f"- **Aquecimento:** {arg.get('aquecimento')} passada(s) com 1 prompt de cada classe nos dois cenários, descartadas do CSV (falhas no aquecimento: {meta.get('aquecimento_falhas')})")
        L.append(f"- **Limite de tokens (`OLLAMA_NUM_PREDICT`):** {cfg.get('num_predict') or 'SEM LIMITE'} (mesmo valor no balancer e no baseline, lido do `.env`)")
        L.append(f"- Timeout do cliente: {cfg.get('timeout_cliente_s')} s | baseline: `{cfg.get('modelo_baseline')}` direto no Ollama")
        L.append(f"- Ollama {meta.get('ollama_versao')} | modelos instalados: {', '.join(meta.get('modelos_instalados', []))}")
        L.append(f"- Modelos carregados após o aquecimento: {', '.join(carregados) or 'nenhum'}")
        L.append(f"- Máquina: {amb.get('so')} | CPU: {amb.get('cpu')} ({amb.get('cpu_nucleos')} núcleos) | RAM: {amb.get('ram_total_gb')} GB | GPU: {amb.get('gpu')}")
        L.append(f"- Código: branch `{amb.get('git_branch')}`, commit `{amb.get('git_commit')}` | Python {amb.get('python')}")
    else:
        L.append("- `metadados.json` não encontrado.")
    L.append("\n## 2. Amostra\n")
    L.append("| Cenário | Linhas | Sucesso | Falhas |\n|---|---:|---:|---:|")
    for c in CENARIOS:
        L.append(f"| {c} | {total[c]} | {res[c]['n']} | {falhas[c]} |")
    L.append("\nMédia, mediana, P95, P99 e máximo usam apenas requisições com sucesso.\n")
    L.append("## 3. Latência (s)\n")
    L.append("| Cenário | n | Média | Mediana | Desv. padrão | P95 | P99 | Máx |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in CENARIOS:
        m = res[c]
        L.append(f"| {c} | {m['n']} | {m['media']:.2f} | {m['mediana']:.2f} | {m['dp']:.2f} | {m['p95']:.2f} | {m['p99']:.2f} | {m['max']:.2f} |")
    L.append(f"\n- Variação da latência média (proposta vs. baseline): **{-red:+.1%}**")
    if difs:
        L.append(f"- Comparação pareada: a proposta foi mais rápida em **{mais_rapidos} de {len(difs)}** prompts (média por prompt nas repetições)")
    if overhead:
        L.append(f"- Custo da camada (tempo no cliente menos `e2e` do balancer, só proposta): média {np.mean(overhead) * 1000:.0f} ms")
    if deriva:
        L.append("\n**Verificação de viés de ordem/deriva** (cada prompt roda uma vez em cada cenário por rodada; "
                 "quem vai primeiro é balanceado por classe):\n")
        L.append("| Cenário | Média passo 1 (s) | Média passo 2 (s) | Variação ajustada por classe | n passo 1 | n passo 2 |\n|---|---:|---:|---:|---:|---:|")
        for c in CENARIOS:
            if c in deriva:
                m1, m2, n1, n2, adj = deriva[c]
                L.append(f"| {c} | {m1:.2f} | {m2:.2f} | {adj:+.1%} | {n1} | {n2} |")
        L.append("\nAs médias gerais dos passos podem diferir só porque a mistura de classes não é idêntica entre eles; "
                 "a variação ajustada compara cada classe com ela mesma e é a que indica deriva.")
    L.append("\n## 4. Por classe de prompt (latência média, s)\n")
    L.append("| Classe | Baseline | Proposta | n baseline | n proposta | Chars resp. baseline | Chars resp. proposta |\n|---|---:|---:|---:|---:|---:|---:|")
    for k in CLASSES:
        b, p = lat_cl.get(("baseline", k), []), lat_cl.get(("proposta", k), [])
        cb, cp = chars_cl.get(("baseline", k), []), chars_cl.get(("proposta", k), [])
        L.append(f"| {NOME_CLASSE[k]} | {np.mean(b):.2f} | {np.mean(p):.2f} | {len(b)} | {len(p)} | "
                 f"{np.mean(cb) if cb else 0:.0f} | {np.mean(cp) if cp else 0:.0f} |")
    L.append("\n## 5. Roteamento da proposta\n")
    L.append(f"- Classe atribuída = classe esperada: **{certos} de {len(prop)} ({certos / len(prop):.1%})**. "
             "Atenção: o conjunto de prompts é fácil para um classificador por regras; isso não mede acurácia geral.")
    L.append(f"- Desvios para o fallback: **{desvios} de {len(prop)}**")
    L.append("- Atendido por: " + ", ".join(f"`{k}`: {v}" for k, v in sorted(atendido.items())))
    if erradas:
        L.append("- Prompts classificados diferente do esperado: " + "; ".join(f"{i} ({e}→{o})" for i, e, o in erradas))
    L.append("\n## 6. Avisos automáticos\n")
    L.extend([f"- {w}" for w in avisos] or ["- Nenhum aviso."])
    L.append("\n## 7. Tabela em LaTeX (números com vírgula)\n")
    L.append("```latex")
    L.append("\\begin{table}[ht]\n\\centering\n\\caption{Latência (s) por cenário na amostra preliminar.}\n\\label{tab:prelim}")
    L.append("\\begin{tabular}{lrrrr}\n\\hline\nCenário & n & Média & P95 & P99 \\\\\n\\hline")
    for c in CENARIOS:
        m = res[c]
        L.append(f"{c.capitalize()} & {m['n']} & {br(m['media'])} & {br(m['p95'])} & {br(m['p99'])} \\\\")
    L.append("\\hline\n\\end{tabular}\n\\end{table}")
    L.append("```")
    rel_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Relatório em {rel_path}")


if __name__ == "__main__":
    main()
