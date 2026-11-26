"""Gera um conjunto grande e variado de prompts rotulados usando um LLM no Ollama.

O conjunto serve para duas coisas:
  1. treinar e medir a acurácia do classificador (não precisa rodar os modelos do balancer);
  2. montar um subconjunto estratificado para o experimento de latência (baseline x proposta).

Como funciona:
  - Uma matriz de variação define as "células": classe x subtipo x idioma (x linguagem de
    programação e dificuldade, quando fazem sentido). Cada célula vira uma chamada ao LLM
    gerador, que devolve um lote de prompts em JSON.
  - As respostas brutas vão para dados/prompts_gerados_bruto.jsonl, uma linha por célula.
    O script é retomável: rodar de novo pula as células que já estão no arquivo.
  - A finalização limpa o texto, remove duplicatas e quase-duplicatas, separa treino/teste
    de forma estratificada e grava:
        prompts_gerados.json      -> todos os prompts com metadados e o campo "split"
        prompts_experimento.json  -> os 40 prompts escritos à mão + N do split de teste
        prompts_gerados.metadados.json -> modelo gerador, seed, contagens, template

O rótulo (classe_esperada) é a classe pedida ao gerador. Ele pode errar: revise uma amostra
(--amostra-revisao) e liste os ids descartados em dados/descartados.json.

Uso:
  python -m src generate-prompts                      # gera (retomável) e finaliza
  python -m src generate-prompts --modelo qwen2.5:7b  # outro gerador
  python -m src generate-prompts --so-finalizar       # só refaz limpeza/split/subconjunto
  python -m src generate-prompts --listar-celulas     # mostra a matriz sem chamar o LLM
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
load_dotenv(BASE.parent / ".env")

OLLAMA = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
MODELO_GERADOR = os.getenv("MODELO_GERADOR", "gemma4:e4b-it-qat")

DADOS = BASE / "dados"
BRUTO = DADOS / "prompts_gerados_bruto.jsonl"
DESCARTADOS = DADOS / "descartados.json"
SAIDA = BASE / "prompts_gerados.json"
SAIDA_EXPERIMENTO = BASE / "prompts_experimento.json"
GOLDEN = BASE / "prompts.json"

IDIOMAS = {"pt": "português do Brasil", "en": "inglês", "es": "espanhol"}
LINGUAGENS = ["Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#", "Go", "Rust", "SQL", "Bash", "PHP"]
DIFICULDADES = ["fácil", "média", "difícil"]

# --------------------------------------------------------------------------- #
# Matriz de variação
# --------------------------------------------------------------------------- #
# Cada subtipo: descrição para o gerador, tamanho do lote, se usa linguagem de programação,
# quantas chamadas por idioma (para variar linguagem/dificuldade) e se é "caso difícil" para
# o classificador por regras (adversarial).
DEFINICAO_CLASSE = {
    "simples": ("pedidos curtos e diretos, com resposta curta e factual, que até um modelo muito "
                "pequeno (150M parâmetros) responderia; sem raciocínio em várias etapas e sem código"),
    "codigo": ("pedidos de programação: escrever, corrigir, explicar, testar ou refatorar código; "
               "um modelo especializado em código é o destino certo"),
    "complexo": ("pedidos que exigem raciocínio em várias etapas, análise, argumentação, "
                 "demonstração ou processamento de um texto longo; um modelo pequeno erraria"),
}

SUBTIPOS = {
    "simples": {
        "factual": dict(desc="pergunta factual de conhecimento geral (geografia, história, ciência, cultura)", lote=10),
        "traducao": dict(desc="pedido para traduzir uma palavra ou frase curta, que deve vir escrita no prompt", lote=10),
        "conversao": dict(desc="conversão de unidades, moedas fictícias com taxa dada, datas ou formatos", lote=10),
        "aritmetica": dict(desc="conta aritmética de uma ou duas operações, ou problema de uma etapa", lote=10),
        "sentimento": dict(desc="classificar o sentimento ou o tom de uma frase curta que vem no prompt", lote=10),
        "gramatica": dict(desc="plural, sinônimo, antônimo, ortografia, classe gramatical de uma palavra", lote=10),
        "definicao": dict(desc="pedido de definição curta de um termo comum, em uma frase", lote=10),
        "reescrita": dict(desc="reescrever uma frase curta (mais formal, mais educada, mais curta) que vem no prompt", lote=10),
        "conversa": dict(desc="mensagem casual de conversa: saudação, agradecimento, pergunta pessoal ao assistente", lote=10),
        "termo_tecnico": dict(
            desc=("pergunta factual curta SOBRE tecnologia ou programação que NÃO pede código: história, "
                  "datas, criadores, significado de siglas (ex.: quem criou a linguagem Python, o que significa "
                  "a sigla API, em que ano saiu o Java). Use palavras como Python, Java, API, SQL, HTML"),
            lote=10, adversarial=True),
    },
    "codigo": {
        "escrever_funcao": dict(desc="escrever uma função ou classe com especificação e exemplo de entrada/saída",
                                lote=8, usa_linguagem=True, chamadas=2),
        "debug": dict(desc=("corrigir um erro: o prompt DEVE conter o trecho de código (5 a 15 linhas) E a mensagem "
                            "de erro ou o comportamento errado, escritos por extenso"),
                      lote=4, usa_linguagem=True, chamadas=2),
        "refatorar": dict(desc="refatorar ou otimizar um trecho de código que DEVE vir escrito no prompt",
                          lote=4, usa_linguagem=True, chamadas=2),
        "explicar_codigo": dict(desc="explicar o que faz um trecho de código que DEVE vir escrito no prompt",
                                lote=4, usa_linguagem=True, chamadas=2),
        "consulta_sql_regex": dict(desc="escrever uma consulta SQL ou uma expressão regular para um caso descrito",
                                   lote=8, chamadas=1),
        "testes": dict(desc="escrever testes unitários para uma função descrita ou mostrada no prompt",
                       lote=6, usa_linguagem=True, chamadas=1),
        "ferramentas": dict(desc="comando de terminal, git, Docker, configuração de build ou de CI",
                            lote=8, chamadas=1),
        "sem_palavra_chave": dict(
            desc=("pedido de programação escrito em linguagem natural SEM usar as palavras código, função, "
                  "programa, script, implementar, nem o nome de linguagens (ex.: 'como faço pra ordenar uma lista "
                  "de dicionários pela idade?', 'qual o jeito certo de ler um arquivo linha por linha?')"),
            lote=8, chamadas=2, adversarial=True),
    },
    "complexo": {
        "prova": dict(desc="demonstração matemática ou prova formal", lote=8),
        "logica": dict(desc="problema de lógica ou quebra-cabeça com várias restrições, enunciado completo", lote=6),
        "multietapas": dict(desc="problema com números que exige várias etapas de cálculo e planejamento", lote=6),
        "comparacao": dict(desc="comparação analítica de duas ou mais alternativas com critérios", lote=8),
        "argumentacao": dict(desc="dilema ético, ensaio argumentativo ou análise crítica de uma posição", lote=8),
        "explicacao_profunda": dict(desc="explicação científica ou técnica aprofundada de um fenômeno, com causas", lote=8),
        "planejamento": dict(desc="planejamento detalhado com restrições (cronograma, orçamento, estratégia)", lote=6),
        "resumo_texto": dict(desc=("resumir ou analisar um texto que DEVE vir no prompt, com 120 a 200 palavras "
                                   "(notícia, relato, trecho de artigo)"), lote=3),
        "curto_dificil": dict(
            desc=("pergunta CURTA (até 15 palavras) mas difícil, sem palavras como explique, analise, compare, "
                  "por que (ex.: 'Existem infinitos números primos gêmeos?', 'Qual a melhor jogada de abertura no "
                  "xadrez e quais as consequências?')"),
            lote=10, adversarial=True),
    },
}

TEMPLATE = """Você está criando um conjunto de dados para treinar um classificador que roteia pedidos
de usuários para modelos de IA de portes diferentes.

Classe deste lote: "{classe}" = {definicao_classe}.
Subtipo: {subtipo_desc}.
{extras}
Gere exatamente {lote} prompts DIFERENTES entre si, como um usuário real os escreveria a um assistente.
Regras:
- Escreva todos os prompts em {idioma}.
- Cada prompt deve ser autocontido: se mencionar um código, erro ou texto, inclua-o por extenso no
  próprio prompt (nunca escreva "segue o código" sem o código).
- Varie o tema, o tom (formal, informal, com erros de digitação ocasionais) e o tamanho.
- Não numere os prompts e não mencione a classe nem o subtipo.
Responda somente com JSON no formato {{"prompts": ["...", "..."]}}."""

SCHEMA = {"type": "object", "properties": {"prompts": {"type": "array", "items": {"type": "string"}}},
          "required": ["prompts"]}


def montar_celulas(seed: int):
    """Lista determinística de células (cada uma = uma chamada ao gerador)."""
    rng = random.Random(seed)
    celulas = []
    for classe, subtipos in SUBTIPOS.items():
        for subtipo, cfg in subtipos.items():
            for idioma in IDIOMAS:
                for k in range(cfg.get("chamadas", 1)):
                    linguagem = rng.choice(LINGUAGENS) if cfg.get("usa_linguagem") else ""
                    dificuldade = rng.choice(DIFICULDADES) if classe != "simples" else "fácil"
                    chave = f"{classe}|{subtipo}|{idioma}|{k}"
                    celulas.append({"chave": chave, "classe": classe, "subtipo": subtipo, "idioma": idioma,
                                    "linguagem": linguagem, "dificuldade": dificuldade,
                                    "lote": cfg["lote"], "adversarial": bool(cfg.get("adversarial"))})
    return celulas


def montar_prompt_gerador(c) -> str:
    extras = []
    if c["linguagem"]:
        extras.append(f"Linguagem de programação: {c['linguagem']}.")
    if c["classe"] != "simples":
        extras.append(f"Dificuldade: {c['dificuldade']}.")
    return TEMPLATE.format(classe=c["classe"], definicao_classe=DEFINICAO_CLASSE[c["classe"]],
                           subtipo_desc=SUBTIPOS[c["classe"]][c["subtipo"]]["desc"],
                           extras="\n".join(extras), lote=c["lote"], idioma=IDIOMAS[c["idioma"]])


MINIMO_LOTE = 0.6  # fração do lote abaixo da qual a célula é gerada de novo


def gerar_lote(client: httpx.Client, modelo: str, c, seed: int, temperatura: float):
    corpo = {"model": modelo, "prompt": montar_prompt_gerador(c), "stream": False, "format": SCHEMA,
             "think": False,
             "options": {"temperature": temperatura, "seed": seed, "num_ctx": 8192, "num_predict": 4096}}
    r = client.post(f"{OLLAMA}/api/generate", json=corpo)
    r.raise_for_status()
    d = r.json()
    prompts = json.loads(d.get("response") or "{}").get("prompts", [])
    if not isinstance(prompts, list):
        raise ValueError("campo 'prompts' não é lista")
    return [p for p in prompts if isinstance(p, str)], d


# --------------------------------------------------------------------------- #
# Limpeza, deduplicação e split
# --------------------------------------------------------------------------- #
_NUMERACAO = re.compile(r"^\s*(?:\d+[\.\)]|[-*•])\s+")


def limpar(texto: str) -> str:
    t = _NUMERACAO.sub("", texto.strip())
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'“”":
        t = t[1:-1].strip()
    return t


def chave_normalizada(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"\W+", " ", t).strip()


def remover_quase_duplicatas(itens, limiar: float):
    """Remove itens com similaridade de cosseno (TF-IDF de caracteres) >= limiar a um item anterior."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("AVISO: scikit-learn ausente; só duplicatas exatas foram removidas.")
        return itens, 0
    if len(itens) < 2:
        return itens, 0
    textos = [chave_normalizada(i["prompt"]) for i in itens]
    m = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(textos)
    sim = cosine_similarity(m, dense_output=False).tolil()
    manter, removidos = [], set()
    for i in range(len(itens)):
        if i in removidos:
            continue
        manter.append(itens[i])
        for j, v in zip(sim.rows[i], sim.data[i]):
            if j > i and v >= limiar:
                removidos.add(j)
    return manter, len(removidos)


def separar(itens, frac_teste: float, seed: int):
    """Split estratificado por (classe, subtipo, idioma)."""
    rng = random.Random(seed)
    grupos = defaultdict(list)
    for it in itens:
        grupos[(it["classe_esperada"], it["subtipo"], it["idioma"])].append(it)
    for chave in sorted(grupos):
        g = grupos[chave]
        rng.shuffle(g)
        n_teste = max(1, round(len(g) * frac_teste)) if len(g) > 1 else 0
        for k, it in enumerate(g):
            it["split"] = "teste" if k < n_teste else "treino"


def subconjunto(itens, por_classe: int, seed: int):
    """Escolhe `por_classe` prompts de teste por classe, alternando subtipo e idioma."""
    rng = random.Random(seed)
    escolhidos = []
    for classe in ("simples", "codigo", "complexo"):
        grupos = defaultdict(list)
        for it in itens:
            if it["split"] == "teste" and it["classe_esperada"] == classe:
                grupos[(it["subtipo"], it["idioma"])].append(it)
        for g in grupos.values():
            rng.shuffle(g)
        chaves = sorted(grupos)
        rng.shuffle(chaves)
        sel = []
        while len(sel) < por_classe and any(grupos[k] for k in chaves):
            for k in chaves:
                if grupos[k] and len(sel) < por_classe:
                    sel.append(grupos[k].pop())
        escolhidos.extend(sel)
    return escolhidos


def finalizar(a):
    if not BRUTO.exists():
        sys.exit(f"ERRO: {BRUTO} não existe. Rode a geração primeiro.")
    linhas = [json.loads(l) for l in BRUTO.read_text(encoding="utf-8").splitlines() if l.strip()]
    descartados = set(json.loads(DESCARTADOS.read_text(encoding="utf-8"))) if DESCARTADOS.exists() else set()
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    vistos = {chave_normalizada(g["prompt"]) for g in golden}

    itens, n_bruto, n_curto, n_dup = [], 0, 0, 0
    contador = Counter()
    letra = {"simples": "S", "codigo": "C", "complexo": "X"}
    for l in linhas:
        c = l["celula"]
        for p in l["prompts"]:
            n_bruto += 1
            texto = limpar(p)
            if len(texto.split()) < 2:
                n_curto += 1
                continue
            k = chave_normalizada(texto)
            if k in vistos:
                n_dup += 1
                continue
            vistos.add(k)
            # id estável: depende do conteúdo, não da ordem de geração
            h = hashlib.sha1(k.encode("utf-8")).hexdigest()[:6]
            pid = f"G{letra[c['classe']]}-{h}"
            contador[pid] += 1
            if contador[pid] > 1:
                pid = f"{pid}-{contador[pid]}"
            itens.append({"id": pid, "classe_esperada": c["classe"], "prompt": texto, "subtipo": c["subtipo"],
                          "idioma": c["idioma"], "linguagem": c["linguagem"], "dificuldade": c["dificuldade"],
                          "adversarial": c["adversarial"], "gerado_por": l["modelo"]})

    itens, n_quase = remover_quase_duplicatas(itens, a.limiar_similaridade)
    n_desc = sum(1 for it in itens if it["id"] in descartados)
    itens = [it for it in itens if it["id"] not in descartados]
    separar(itens, a.frac_teste, a.seed)
    itens.sort(key=lambda it: (it["classe_esperada"], it["subtipo"], it["idioma"], it["id"]))
    SAIDA.write_text(json.dumps(itens, ensure_ascii=False, indent=1), encoding="utf-8")

    sub = subconjunto(itens, a.subconjunto_por_classe, a.seed)
    golden_marcado = [{**g, "subtipo": "manual", "idioma": "pt", "linguagem": "", "dificuldade": "",
                       "adversarial": False, "gerado_por": "manual"} for g in golden]
    SAIDA_EXPERIMENTO.write_text(json.dumps(golden_marcado + sub, ensure_ascii=False, indent=1), encoding="utf-8")

    por = lambda campo: dict(sorted(Counter(it[campo] for it in itens).items()))
    meta = {
        "finalizado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modelos_geradores": sorted({l["modelo"] for l in linhas}),
        "celulas": len(linhas),
        "brutos": n_bruto, "descartados_curtos": n_curto, "duplicatas_exatas": n_dup,
        "quase_duplicatas": n_quase, "limiar_similaridade": a.limiar_similaridade,
        "descartados_revisao": n_desc, "finais": len(itens),
        "por_classe": por("classe_esperada"), "por_idioma": por("idioma"), "por_subtipo": por("subtipo"),
        "por_split": por("split"), "adversariais": sum(it["adversarial"] for it in itens),
        "seed": a.seed, "frac_teste": a.frac_teste,
        "subconjunto_experimento": {"golden": len(golden), "gerados": len(sub),
                                    "por_classe": dict(Counter(it["classe_esperada"] for it in sub))},
        "template": TEMPLATE,
    }
    SAIDA.with_name("prompts_gerados.metadados.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Brutos: {n_bruto} | curtos: {n_curto} | duplicatas: {n_dup} | quase-duplicatas: {n_quase} | "
          f"descartados na revisão: {n_desc} | finais: {len(itens)}")
    print("Por classe:", meta["por_classe"], "| por split:", meta["por_split"])
    print(f"Gravado: {SAIDA.name} e {SAIDA_EXPERIMENTO.name} ({len(golden)} manuais + {len(sub)} gerados)")

    if a.amostra_revisao:
        rng = random.Random(a.seed)
        print(f"\nAmostra para revisão manual ({a.amostra_revisao}):")
        for it in rng.sample(itens, min(a.amostra_revisao, len(itens))):
            print(f"- [{it['id']}] {it['classe_esperada']}/{it['subtipo']}/{it['idioma']}: {it['prompt'][:160]}")


def gerar(a):
    DADOS.mkdir(parents=True, exist_ok=True)
    feitas = set()
    if BRUTO.exists():
        feitas = {json.loads(l)["celula"]["chave"] for l in BRUTO.read_text(encoding="utf-8").splitlines() if l.strip()}
    celulas = [c for c in montar_celulas(a.seed) if c["chave"] not in feitas]
    if a.limite:
        celulas = celulas[: a.limite]
    print(f"Gerador: {a.modelo} em {OLLAMA} | células pendentes: {len(celulas)} (já feitas: {len(feitas)})")
    falhas = 0
    with httpx.Client(timeout=a.timeout) as client, open(BRUTO, "a", encoding="utf-8") as f:
        for i, c in enumerate(celulas, 1):
            t0 = time.perf_counter()
            melhor, d, erro = None, {}, ""
            for tentativa in range(1, a.tentativas + 1):
                try:
                    prompts, resp = gerar_lote(client, a.modelo, c, a.seed + tentativa, a.temperatura)
                except Exception as e:  # JSON inválido, timeout etc.: tenta de novo
                    erro = f"{type(e).__name__}: {e}"
                    continue
                if melhor is None or len(prompts) > len(melhor):
                    melhor, d = prompts, resp
                # Às vezes o gerador junta vários prompts numa string só e devolve 1 ou 2 itens.
                # Lote incompleto conta como tentativa falha; fica o melhor resultado obtido.
                if len(prompts) >= MINIMO_LOTE * c["lote"]:
                    break
                erro = f"lote incompleto ({len(prompts)}/{c['lote']})"
            if melhor is None:
                falhas += 1
                print(f"[{i}/{len(celulas)}] {c['chave']} FALHOU: {erro}")
                continue
            prompts = melhor
            f.write(json.dumps({"celula": c, "modelo": a.modelo, "prompts": prompts,
                                "eval_count": d.get("eval_count"),
                                "gerado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                               ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(celulas)}] {c['chave']:40s} {len(prompts):2d} prompts {time.perf_counter() - t0:6.1f}s",
                  flush=True)
    print(f"Geração concluída; falhas: {falhas}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=MODELO_GERADOR, help="tag do Ollama usada para gerar")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperatura", type=float, default=0.9)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--tentativas", type=int, default=3)
    ap.add_argument("--limite", type=int, default=0, help="gera no máximo N células nesta execução (0 = todas)")
    ap.add_argument("--frac-teste", type=float, default=0.3)
    ap.add_argument("--limiar-similaridade", type=float, default=0.9)
    ap.add_argument("--subconjunto-por-classe", type=int, default=40,
                    help="prompts gerados por classe no prompts_experimento.json")
    ap.add_argument("--amostra-revisao", type=int, default=0, help="imprime N prompts aleatórios para revisão")
    ap.add_argument("--so-finalizar", action="store_true", help="não chama o LLM; só limpa/separa/grava")
    ap.add_argument("--listar-celulas", action="store_true")
    a = ap.parse_args()

    if a.listar_celulas:
        cel = montar_celulas(a.seed)
        print(f"{len(cel)} células, ~{sum(c['lote'] for c in cel)} prompts")
        for c in cel:
            print(f"  {c['chave']:40s} lote={c['lote']:2d} {c['linguagem']:10s} {c['dificuldade']}")
        return
    if not a.so_finalizar:
        gerar(a)
    finalizar(a)


if __name__ == "__main__":
    main()
