"""Lê resultados.csv (v2 ou legado), agrega por matriz e gera relatório.

Saídas: ``resultados_resumo.csv``, ``relatorio_resultados.md`` e gráficos em
``graficos/``. As agregações puras ficam em ``analise_metricas.py``; este
arquivo só faz IO, console, Markdown e plots (matplotlib é importado sob
demanda, então a análise funciona sem o extra de gráficos).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_RAIZ = Path(__file__).resolve().parent.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from experiments.analise_metricas import (  # noqa: E402
    CENARIO_LB, CENARIO_PADRAO, CENARIOS, CLASSES, COLUNAS_RESUMO,
    agregar, avisos, normalizar_cenario, numero, sucesso, valor,
)

NOME_CLASSE = {"simples": "Simples", "codigo": "Código", "complexo": "Complexo"}


def carregar(caminho):
    with open(caminho, newline="", encoding="utf-8") as arquivo:
        return list(csv.DictReader(arquivo))


def br(x, casas=2):
    return "—" if x is None else f"{float(x):.{casas}f}".replace(".", ",")


def media(valores):
    return sum(valores) / len(valores) if valores else None


def chaves_ordenadas(grupos):
    """Combinações primeiro, subtotais por cenário e total por último."""
    def prioridade(chave):
        cenario, deployment, classe = chave
        return (
            0 if cenario != "todos" and deployment != "todos" and classe != "todos" else
            1 if cenario != "todos" and classe == "todos" else 2,
            cenario, deployment, classe,
        )
    return sorted((c for c in grupos if c != ("todos", "todos", "todos")), key=prioridade)


def escrever_resumo(grupos, destino):
    with open(destino, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS_RESUMO, lineterminator="\n")
        escritor.writeheader()
        for chave in chaves_ordenadas(grupos) + [("todos", "todos", "todos")]:
            linha = grupos[chave]
            escritor.writerow({coluna: linha.get(coluna, "") for coluna in COLUNAS_RESUMO})
    return destino


def _graficos(grupos, pasta):
    """Gera os gráficos mínimos; devolve False quando matplotlib não está instalado."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    combinacoes = [(c, d) for c, d, classe in chaves_ordenadas(grupos)
                   if classe == "todos" and c in CENARIOS and d != "todos"]
    rotulos = [f"{d}/{c}" for c, d in combinacoes]
    if not rotulos:
        return False
    x = list(range(len(combinacoes)))
    cores = {CENARIO_PADRAO: "#8c8c8c", CENARIO_LB: "#1f77b4"}

    def figura(nome, titulo, series, eixo, legenda=False):
        fig, ax = plt.subplots(figsize=(max(6.5, 2.6 * len(x)), 4))
        largura = 0.8 / max(1, len(series))
        for indice, (serie, valores) in enumerate(series.items()):
            deslocamento = (indice - (len(series) - 1) / 2) * largura
            barras = ax.bar([i + deslocamento for i in x], valores, largura,
                            label=serie, color=cores.get(serie, None))
            ax.bar_label(barras, fmt="%.2f", fontsize=7)
        ax.set_xticks(x, rotulos, rotation=20, ha="right")
        ax.set_ylabel(eixo)
        ax.set_title(titulo, pad=18)
        if legenda:
            ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(pasta / nome, dpi=200)
        plt.close(fig)

    def valor_grupo(cenario, deployment, campo):
        return float(grupos.get((cenario, deployment, "todos"), {}).get(campo) or 0)

    figura(
        "grafico1_latencia_geral.png", "Latência por cenário × deployment (s)",
        {"P95": [valor_grupo(c, d, "latency_p95_s") for c, d in combinacoes],
         "P99": [valor_grupo(c, d, "latency_p99_s") for c, d in combinacoes]},
        "Latência (s)", legenda=True,
    )

    fig, ax = plt.subplots(figsize=(max(6.5, 2.6 * len(x)), 4))
    largura = 0.8 / max(1, len(CLASSES))
    for indice, classe in enumerate(CLASSES):
        valores = [float(grupos.get((c, d, classe), {}).get("latency_mean_s") or 0)
                   for c, d in combinacoes]
        barras = ax.bar([i + (indice - 1) * largura for i in x], valores, largura,
                        label=NOME_CLASSE[classe])
        ax.bar_label(barras, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, rotulos, rotation=20, ha="right")
    ax.set_ylabel("Latência média (s)")
    ax.set_title("Latência por classe, cenário e deployment")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(pasta / "grafico2_latencia_por_classe.png", dpi=200)
    plt.close(fig)

    figura(
        "grafico3_recursos.png", "RAM e GPU por cenário × deployment",
        {"RAM RSS (MB)": [valor_grupo(c, d, "ram_rss_mb_avg") for c, d in combinacoes],
         "GPU mem (MB)": [valor_grupo(c, d, "gpu_mem_used_mb_avg") for c, d in combinacoes]},
        "Média por bloco", legenda=True,
    )
    if any(grupos.get((c, d, "todos"), {}).get("token_rows") for c, d in combinacoes):
        figura(
            "grafico4_tokens.png", "Uso de tokens por cenário × deployment",
            {"Prompt (média)": [valor_grupo(c, d, "prompt_tokens_avg") for c, d in combinacoes],
             "Completion (média)": [valor_grupo(c, d, "completion_tokens_avg") for c, d in combinacoes]},
            "Tokens por requisição", legenda=True,
        )
    if any(c == CENARIO_LB and grupos.get((c, d, "todos"), {}).get("n_classified")
           for c, d in combinacoes):
        figura(
            "grafico5_erro_classificacao.png", "Taxa de erro de classificação (loss)",
            {"Erro": [float(grupos.get((c, d, "todos"), {}).get("classification_error_rate") or 0)
                      if c == CENARIO_LB else 0 for c, d in combinacoes]},
            "Taxa de erro", 
        )
    return True


def _dados_comparacao(linhas):
    """Comparações pareadas e contagens no estilo do relatório antigo."""
    por_prompt = defaultdict(lambda: defaultdict(list))
    chars_cl = defaultdict(list)
    por_passo_cl = defaultdict(list)
    erros = Counter()
    for linha in linhas:
        cenario = normalizar_cenario(valor(linha, "scenario", "cenario"))
        if not sucesso(linha):
            erros[(cenario, valor(linha, "erro").split(":")[0] or "erro")] += 1
            continue
        lat = numero(linha, "latency_e2e_s", "latencia_s")
        if lat is not None:
            por_prompt[valor(linha, "prompt_id")][cenario].append(lat)
            if valor(linha, "passo"):
                por_passo_cl[(cenario, valor(linha, "passo"), valor(linha, "classe_esperada"))].append(lat)
        chars = numero(linha, "resposta_chars")
        if chars is not None:
            chars_cl[(cenario, valor(linha, "classe_esperada"))].append(chars)

    difs = [
        media(p[CENARIO_PADRAO]) - media(p[CENARIO_LB])
        for p in por_prompt.values() if p[CENARIO_PADRAO] and p[CENARIO_LB]
    ]
    return {
        "difs": difs,
        "mais_rapidos": sum(1 for d in difs if d > 0),
        "chars_cl": chars_cl,
        "por_passo_cl": por_passo_cl,
        "erros": erros,
    }


def _deriva(dados):
    resultado = {}
    for cenario in CENARIOS:
        razoes = []
        for classe in CLASSES:
            p1 = dados["por_passo_cl"].get((cenario, "1", classe), [])
            p2 = dados["por_passo_cl"].get((cenario, "2", classe), [])
            if p1 and p2 and media(p1):
                razoes.append(media(p2) / media(p1))
        if razoes:
            resultado[cenario] = media(razoes) - 1
    return resultado


def main():
    ap = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent
    ap.add_argument("--entrada", default=str(base / "resultados.csv"))
    ap.add_argument("--pasta", default=str(base / "graficos"))
    ap.add_argument("--relatorio", default=None)
    ap.add_argument("--resumo", default=None, help="CSV de agregações (padrão: resultados_resumo.csv)")
    args = ap.parse_args()

    entrada = Path(args.entrada)
    pasta = Path(args.pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    relatorio = Path(args.relatorio) if args.relatorio else entrada.with_name("relatorio_resultados.md")
    resumo = Path(args.resumo) if args.resumo else entrada.with_name("resultados_resumo.csv")
    meta_path = entrada.with_name(entrada.stem + ".metadados.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    linhas = carregar(entrada)
    grupos = agregar(linhas)
    total = grupos.get(("todos", "todos", "todos"), {})
    if not total.get("ok"):
        print(f"Sem nenhuma requisição bem-sucedida. Total/falhas: "
              f"{total.get('n', 0)} / {total.get('fail', 0)}.")
        raise SystemExit(1)

    escrever_resumo(grupos, resumo)
    com_graficos = _graficos(grupos, pasta)
    dados = _dados_comparacao(linhas)
    deriva = _deriva(dados)
    cfg = meta.get("config", {}) or {}
    amb = meta.get("ambiente", {}) or {}
    argumentos = meta.get("argumentos", {}) or {}

    aviso_lista = avisos(linhas, meta, grupos)
    if total.get("fail"):
        aviso_lista.append(f"Tipos de erro: {dict(dados['erros'])}.")
    if deriva:
        for cenario, ajuste in deriva.items():
            if abs(ajuste) > 0.15:
                aviso_lista.append(
                    f"Deriva entre passos em {cenario}: {ajuste:+.0%} (média das variações dentro de cada classe)."
                )
    if not com_graficos:
        aviso_lista.append("matplotlib não instalado: gráficos não foram gerados (instale o extra experiments).")

    # ---------- console ----------
    print(f"{'deployment/cenario':30s} {'n':>5s} {'ok':>5s} {'falhas':>6s} "
          f"{'média':>8s} {'P95':>8s} {'P99':>8s} {'erro cls':>9s}")
    for chave in chaves_ordenadas(grupos):
        cenario, deployment, classe = chave
        if classe != "todos":
            continue
        g = grupos[chave]
        print(f"{deployment + '/' + cenario:30s} {g['n']:5d} {g['ok']:5d} {g['fail']:6d} "
              f"{g['latency_mean_s']:8.2f} {g['latency_p95_s']:8.2f} {g['latency_p99_s']:8.2f} "
              f"{br(g['classification_error_rate'], 3) if g['n_classified'] else 'NA':>9s}")
    print(f"Resumo em {resumo}; gráficos em {pasta}")

    # ---------- relatório ----------
    L = ["# Relatório da rodada de experimentos", ""]
    L.append("> Gerado por `analisar.py`. Latência é ponta a ponta no cliente; "
             "`loss` significa taxa de erro de classificação, não loss de treino.\n")
    L.append("## 1. Configuração e ambiente\n")
    if meta:
        L.append(f"- Início/fim (UTC): {meta.get('inicio_utc')} / {meta.get('fim_utc')} | run_id: `{meta.get('run_id')}`")
        L.append(f"- Schema do CSV: {meta.get('schema_version', '1 (legado)')}")
        L.append(f"- Repetições: {argumentos.get('repeticoes')} | prompts: {argumentos.get('n_prompts')} | seed: {argumentos.get('seed')}")
        L.append(f"- Cenários: {', '.join(argumentos.get('scenarios') or [c for c in CENARIOS])} | deployments: "
                 f"{', '.join(argumentos.get('deployments') or ['local'])}")
        L.append(f"- Delineamento: {meta.get('delineamento', 'não registrado')}")
        L.append(f"- Tag comum: `{cfg.get('model_tag') or cfg.get('modelo_baseline')}` | "
                 f"`OLLAMA_NUM_PREDICT`: {cfg.get('num_predict') or 'SEM LIMITE'} | timeout: {cfg.get('timeout_cliente_s')} s")
        L.append(f"- Endpoints: {cfg.get('ollama_local') or cfg.get('ollama_host')} (local), "
                 f"{cfg.get('ollama_remote') or 'não configurado'} (remoto), "
                 f"LB local {cfg.get('lb_local_url') or cfg.get('balancer')}, "
                 f"LB remoto {cfg.get('lb_remote_url') or 'não configurado'}")
        L.append(f"- Amostragem de recursos: {cfg.get('resource_sample_ms', '?')} ms | GPU: {cfg.get('gpu_metrics', '?')}")
        L.append(f"- Máquina: {amb.get('so')} | CPU: {amb.get('cpu')} ({amb.get('cpu_nucleos')} núcleos) | "
                 f"RAM: {amb.get('ram_total_gb')} GB | GPU: {amb.get('gpu')}")
        L.append(f"- Código: `{amb.get('git_branch')}`@{amb.get('git_commit')} | Python {amb.get('python')}")
    else:
        L.append("- `metadados.json` não encontrado.")
    if meta.get("matriz"):
        L.append("\nMatriz executada:\n")
        L.append("| Cenário | Deployment | Ollama efetivo | Tag efetivo | LB efetivo |\n|---|---|---|---|---|")
        for item in meta["matriz"]:
            L.append(f"| {item.get('scenario')} | {item.get('deployment')} | "
                     f"{item.get('effective_ollama_host') or item.get('target_ollama_host')} | "
                     f"{item.get('effective_model_tag') or item.get('model_tag')} | "
                     f"{item.get('effective_lb_url') or '—'} |")

    L.append("\n## 2. Amostra\n")
    L.append("| Cenário | Deployment | Linhas | Sucesso | Falhas | n classificado |\n|---|---|---:|---:|---:|---:|")
    for chave in chaves_ordenadas(grupos):
        cenario, deployment, classe = chave
        if classe != "todos":
            continue
        g = grupos[chave]
        L.append(f"| {cenario} | {deployment} | {g['n']} | {g['ok']} | {g['fail']} | {g['n_classified']} |")

    L.append("\n## 3. Latência ponta a ponta (s)\n")
    L.append("| Cenário | Deployment | n | Média | Mediana | P95 | P99 | Máx |\n|---|---|---:|---:|---:|---:|---:|---:|")
    for chave in chaves_ordenadas(grupos):
        cenario, deployment, classe = chave
        if classe != "todos":
            continue
        g = grupos[chave]
        L.append(f"| {cenario} | {deployment} | {g['ok']} | {br(g['latency_mean_s'])} | "
                 f"{br(g['latency_median_s'])} | {br(g['latency_p95_s'])} | "
                 f"{br(g['latency_p99_s'])} | {br(g['latency_max_s'])} |")
    L.append("\n### Por classe de prompt (latência média, s)\n")
    L.append("| Cenário | Deployment | " + " | ".join(NOME_CLASSE[c] for c in CLASSES) + " |\n|---|---|---:|---:|---:|")
    for cenario in CENARIOS:
        for deployment in sorted({d for c, d, _ in grupos if c == cenario and d != "todos"}):
            cols = [br(grupos.get((cenario, deployment, c), {}).get("latency_mean_s")) for c in CLASSES]
            L.append(f"| {cenario} | {deployment} | " + " | ".join(cols) + " |")

    L.append("\n## 4. RAM e GPU (média / pico por bloco)\n")
    L.append("| Cenário | Deployment | RSS MB | RAM sistema MB | RAM % | GPU util % | GPU mem MB | GPU W | amostras GPU |\n"
             "|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for chave in chaves_ordenadas(grupos):
        cenario, deployment, classe = chave
        if classe != "todos":
            continue
        g = grupos[chave]
        L.append(f"| {cenario} | {deployment} | {br(g['ram_rss_mb_avg'])}/{br(g['ram_rss_mb_peak'])} | "
                 f"{br(g['ram_system_used_mb_avg'])}/{br(g['ram_system_used_mb_peak'])} | "
                 f"{br(g['ram_system_pct_avg'])}/{br(g['ram_system_pct_peak'])} | "
                 f"{br(g['gpu_util_pct_avg'])}/{br(g['gpu_util_pct_peak'])} | "
                 f"{br(g['gpu_mem_used_mb_avg'])}/{br(g['gpu_mem_used_mb_peak'])} | "
                 f"{br(g['gpu_power_w_avg'])}/{br(g['gpu_power_w_peak'])} | {g['gpu_sample_count']} |")
    regra = meta.get("regras_medicao", {}).get("gpu")
    if regra:
        L.append(f"\nRegra de agregação GPU: {regra}")

    L.append("\n## 5. Tokens\n")
    if total.get("token_rows"):
        L.append("| Cenário | Deployment | Prompt (média) | Completion (média) | Total (soma) | linhas com uso |\n"
                 "|---|---|---:|---:|---:|---:|")
        for chave in chaves_ordenadas(grupos):
            cenario, deployment, classe = chave
            if classe != "todos":
                continue
            g = grupos[chave]
            if not g["token_rows"]:
                continue
            L.append(f"| {cenario} | {deployment} | {br(g['prompt_tokens_avg'], 1)} | "
                     f"{br(g['completion_tokens_avg'], 1)} | {br(g['total_tokens_sum'], 0)} | {g['token_rows']} |")
    else:
        L.append("Nenhuma resposta trouxe `prompt_eval_count`/`eval_count`.")

    L.append("\n## 6. Loss no teste = taxa de erro de classificação\n")
    L.append("`classification_error = 0` se a classe predita é igual à esperada, `1` se diferente. "
             "No cenário `default_model` não há classificador: `classe_predita` fica vazia e a linha **não entra** "
             "no denominador. Não é loss de treino de LLM.\n")
    L.append("| Cenário | Deployment | Classe | n classificado | Taxa de erro |\n|---|---|---|---:|---:|")
    for chave in chaves_ordenadas(grupos):
        cenario, deployment, classe = chave
        g = grupos[chave]
        if not g["n_classified"] or classe == "todos":
            continue
        L.append(f"| {cenario} | {deployment} | {NOME_CLASSE.get(classe, classe)} | "
                 f"{g['n_classified']} | {br(g['classification_error_rate'], 3)} |")
    if not any(g["n_classified"] for g in grupos.values()):
        L.append("| — | — | nenhuma classificação disponível | 0 | — |")

    L.append("\n## 7. Roteamento da proposta\n")
    proposta_linhas = [r for r in linhas if normalizar_cenario(valor(r, "scenario", "cenario")) == CENARIO_LB
                       and sucesso(r)]
    desvios = sum(1 for r in proposta_linhas if valor(r, "desvio").lower() == "true")
    atendido = Counter(valor(r, "modelo_atendeu") for r in proposta_linhas)
    L.append(f"- Redirecionamentos ao fallback: **{desvios} de {len(proposta_linhas)}**")
    L.append("- Atendido por: " + (", ".join(f"`{k}`: {v}" for k, v in sorted(atendido.items())) or "—"))
    if dados["difs"]:
        L.append(f"- Comparação pareada: o LB foi mais rápido em {dados['mais_rapidos']} de {len(dados['difs'])} prompts.")
    if deriva:
        L.append("\nVerificação de viés de ordem (variação ajustada por classe, passo 2 vs passo 1): " +
                 ", ".join(f"{c} {a:+.1%}" for c, a in deriva.items()))

    carga = meta.get("carga") or []
    if carga:
        L.append("\n## 8. Vazão sob carga\n")
        L.append("| Rodada | Passo | Cenário | Deployment | Requisições | Duração (s) | Vazão (req/s) |\n"
                 "|---:|---:|---|---|---:|---:|---:|")
        for bloco in carga:
            L.append(f"| {bloco.get('rodada')} | {bloco.get('passo')} | {bloco.get('cenario', '—')} | "
                     f"{bloco.get('deployment', '—')} | {bloco.get('n')} | "
                     f"{br(bloco.get('duracao_s'))} | {br(bloco.get('vazao_rps'), 3)} |")

    L.append("\n## 9. Avisos automáticos\n")
    L.extend([f"- {aviso}" for aviso in aviso_lista] or ["- Nenhum aviso."])

    L.append("\n## 10. Tabela em LaTeX\n")
    L.append("```latex")
    L.append("\\begin{table}[ht]\n\\centering\n\\caption{Latência (s) por cenário e deployment.}\n\\label{tab:matriz}")
    L.append("\\begin{tabular}{llrrrr}\n\\hline\nCenário & Deployment & n & Média & P95 & P99 \\\\\n\\hline")
    for chave in chaves_ordenadas(grupos):
        cenario, deployment, classe = chave
        if classe != "todos":
            continue
        g = grupos[chave]
        L.append(f"{cenario} & {deployment} & {g['ok']} & {br(g['latency_mean_s'])} & "
                 f"{br(g['latency_p95_s'])} & {br(g['latency_p99_s'])} \\\\")
    L.append("\\hline\n\\end{tabular}\n\\end{table}")
    L.append("```")

    relatorio.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Relatório em {relatorio}")


if __name__ == "__main__":
    main()
