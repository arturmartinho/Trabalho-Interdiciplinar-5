"""CLI unificada do projeto.

Camadas:
  * apresentação  -> este módulo
  * aplicação     -> ``src/loadbalancer`` e ``src/classificador.py``
  * domínio       -> ``src/ai_model.py``
  * dados         -> ``src/data_access.py`` (SQLite)

Uso:
    python -m src --help
    python -m src serve
    python -m src models register --name phi3.5 --role complexo --status up
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from src import log
from src.ai_model import AIModelDM, ROLES
from src.loadbalancer.settings import API_HOST, API_PORT

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def _print_models(rows: list[dict]) -> None:
    if not rows:
        log.info("Nenhum modelo registrado. Use 'models register', 'models host' ou 'models connect'.")
        return
    for row in rows:
        print(f"  - {AIModelDM(**row).display()}")


def _cmd_models_list(args: argparse.Namespace) -> int:
    rows = AIModelDM.list()
    if getattr(args, "json", False):
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        _print_models(rows)
    return 0


def _cmd_models_register(args: argparse.Namespace) -> int:
    return 0 if AIModelDM.register(
        args.name,
        type_=args.type,
        url=args.url or "",
        role=args.role,
        status=args.status,
    ) else 1


def _cmd_models_host(args: argparse.Namespace) -> int:
    return 0 if AIModelDM.host(args.name, role=args.role) else 1


def _cmd_models_connect(args: argparse.Namespace) -> int:
    return 0 if AIModelDM.connect(args.name, args.url, role=args.role) else 1


def _resolve_remote_row(args: argparse.Namespace) -> dict | None:
    row = AIModelDM.get(id=args.id, name=args.name)
    if row is None:
        log.error("Modelo não encontrado. Informe --id ou --name.")
    return row


def _cmd_models_remove(args: argparse.Namespace) -> int:
    row = _resolve_remote_row(args)
    if row is None:
        return 1
    AIModelDM.delete({"id": row["id"]})
    log.success(f"Modelo '{row['name']}' (id={row['id']}) removido.")
    return 0


def _cmd_models_update(args: argparse.Namespace) -> int:
    _print_models(AIModelDM.update(args.id))
    return 0


def _cmd_models_stop(args: argparse.Namespace) -> int:
    return 0 if AIModelDM.stop(args.id) else 1


def _cmd_models_resume(args: argparse.Namespace) -> int:
    return 0 if AIModelDM.resume(args.id) else 1


def _cmd_models_call(args: argparse.Namespace) -> int:
    row = _resolve_remote_row(args)
    if row is None:
        return 1
    prompt = args.prompt or input("Prompt: ").strip()
    if not prompt:
        log.warning("Prompt vazio; nada foi enviado.")
        return 1
    answer = AIModelDM.call(row["id"], prompt)
    if answer is None:
        return 1
    print("\n--- Resposta ---")
    print(answer)
    return 0


# --------------------------------------------------------------------------- #
# classificador / inferência
# --------------------------------------------------------------------------- #
def _cmd_classify(args: argparse.Namespace) -> int:
    from src.classificador import classificar

    result = classificar(args.prompt)
    if args.json:
        print(json.dumps(result.como_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"classe:  {result.classe}")
        print(f"modelo:  {result.modelo}")
        print(f"palavras:{result.palavras}")
        print(f"motivo:  {result.motivo}")
    return 0


async def _infer_direct(prompt: str, prompt_id: str | None) -> dict:
    from src.loadbalancer.dispatcher import dispatch, resolve_classifier
    from src.loadbalancer.ollama_client import OllamaClient
    from src.loadbalancer import registry
    from src.loadbalancer.settings import OLLAMA_HOST, OLLAMA_TIMEOUT_S

    fallback = registry.fallback_target()
    model_urls = registry.model_urls()
    model_urls.setdefault(fallback["name"], fallback["url"])

    client = OllamaClient(OLLAMA_HOST, OLLAMA_TIMEOUT_S)
    try:
        result = await dispatch(
            prompt,
            classifier=resolve_classifier(),
            client=client,
            prompt_id=prompt_id,
            fallback_model=fallback["name"],
            model_urls=model_urls,
        )
        return result.as_dict()
    finally:
        await client.aclose()


def _infer_http(url: str, prompt: str, prompt_id: str | None) -> dict:
    import httpx

    response = httpx.post(
        f"{url.rstrip('/')}/infer",
        json={"prompt": prompt, "prompt_id": prompt_id},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def _cmd_infer(args: argparse.Namespace) -> int:
    prompt = args.prompt or input("Prompt: ").strip()
    if not prompt:
        log.warning("Prompt vazio; nada foi enviado.")
        return 1

    try:
        if args.url:
            result = _infer_http(args.url, prompt, args.prompt_id)
        else:
            result = asyncio.run(_infer_direct(prompt, args.prompt_id))
    except Exception as exc:
        log.error(f"Falha na inferência: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result.get("response") or "(sem resposta)")
        print(
            "\n"
            f"classe={result.get('label')} "
            f"modelo_escolhido={result.get('chosen_model')} "
            f"modelo_atendido={result.get('served_model')} "
            f"redirecionado={result.get('redirected')} "
            f"latencia_total_s={result.get('e2e_latency_s')}",
            file=sys.stderr,
        )
    return 0 if result.get("ok") else 1


# --------------------------------------------------------------------------- #
# serviço / experimentos
# --------------------------------------------------------------------------- #
def _cmd_serve(args: argparse.Namespace) -> int:
    from src.loadbalancer.api import main as serve

    serve(host=args.host, port=args.port, reload=args.reload)
    return 0


def _run_script(path: Path, extra: Sequence[str]) -> int:
    if not path.exists():
        log.error(f"Script não encontrado: {path}")
        return 1
    return subprocess.call([sys.executable, str(path), *extra])


def _cmd_experiment(args: argparse.Namespace) -> int:
    return _run_script(EXPERIMENTS_DIR / "rodar_experimento.py", args.args)


def _cmd_analyze(args: argparse.Namespace) -> int:
    return _run_script(EXPERIMENTS_DIR / "analisar.py", args.args)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def _add_model_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--id", type=int, help="ID do modelo no banco")
    parser.add_argument("--name", help="Tag/nome do modelo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="CLI do Load Balancer de modelos de IA.",
    )
    parser.add_argument("--version", action="version", version="ai-loadbalancer 0.2.0")
    commands = parser.add_subparsers(dest="command", metavar="comando")

    # models ----------------------------------------------------------------
    models = commands.add_parser("models", help="Gerenciar o catálogo de modelos de IA")
    models.set_defaults(handler=_cmd_models_list)
    model_commands = models.add_subparsers(dest="models_command", metavar="ação")

    p = model_commands.add_parser("list", help="Listar modelos registrados")
    p.add_argument("--json", action="store_true", help="Saída em JSON")
    p.set_defaults(handler=_cmd_models_list)

    p = model_commands.add_parser("register", help="Registrar um modelo sem testar a conexão")
    p.add_argument("--name", required=True, help="Tag do modelo (ex.: phi3.5)")
    p.add_argument("--type", choices=("local", "remote"), default="local")
    p.add_argument("--role", choices=ROLES, default=None, help="Classe usada pelo roteamento")
    p.add_argument("--url", default="", help="URL do backend (vazio = OLLAMA_HOST)")
    p.add_argument("--status", choices=("up", "down"), default="down")
    p.set_defaults(handler=_cmd_models_register)

    p = model_commands.add_parser("host", help="Instalar/baixar e subir um modelo local no Ollama")
    p.add_argument("--name", required=True, help="Tag do modelo")
    p.add_argument("--role", choices=ROLES, default=None, help="Classe usada pelo roteamento")
    p.set_defaults(handler=_cmd_models_host)

    p = model_commands.add_parser("connect", help="Conectar a um modelo remoto e validar a URL")
    p.add_argument("--name", required=True, help="Tag/nome do modelo")
    p.add_argument("--url", required=True, help="URL do backend")
    p.add_argument("--role", choices=ROLES, default=None, help="Classe usada pelo roteamento")
    p.set_defaults(handler=_cmd_models_connect)

    p = model_commands.add_parser("remove", help="Remover um modelo do catálogo")
    _add_model_selection(p)
    p.set_defaults(handler=_cmd_models_remove)

    p = model_commands.add_parser("update", help="Atualizar o status (ping) de um ou todos")
    p.add_argument("--id", type=int, help="ID do modelo (vazio = todos)")
    p.set_defaults(handler=_cmd_models_update)

    p = model_commands.add_parser("stop", help="Descarregar um modelo local (vazio = todos)")
    p.add_argument("--id", type=int)
    p.set_defaults(handler=_cmd_models_stop)

    p = model_commands.add_parser("resume", help="Carregar um modelo local (vazio = todos)")
    p.add_argument("--id", type=int)
    p.set_defaults(handler=_cmd_models_resume)

    p = model_commands.add_parser("call", help="Enviar um prompt direto a um modelo")
    _add_model_selection(p)
    p.add_argument("--prompt", help="Prompt (vazio = solicita no terminal)")
    p.set_defaults(handler=_cmd_models_call)

    # classify --------------------------------------------------------------
    p = commands.add_parser("classify", help="Classificar um prompt e mostrar o modelo escolhido")
    p.add_argument("--prompt", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=_cmd_classify)

    # infer -----------------------------------------------------------------
    p = commands.add_parser("infer", help="Executar uma inferência pelo load balancer")
    p.add_argument("--prompt", help="Prompt (vazio = solicita no terminal)")
    p.add_argument("--prompt-id", help="Identificador do prompt")
    p.add_argument("--url", help="Se informado, chama um load balancer já em execução")
    p.add_argument("--json", action="store_true", help="Saída completa em JSON")
    p.set_defaults(handler=_cmd_infer)

    # serve -----------------------------------------------------------------
    p = commands.add_parser("serve", help="Subir a API HTTP do load balancer")
    p.add_argument("--host", default=API_HOST)
    p.add_argument("--port", type=int, default=API_PORT)
    p.add_argument("--reload", action="store_true", help="Recarregar ao alterar arquivos")
    p.set_defaults(handler=_cmd_serve)

    # experiment / analyze --------------------------------------------------
    p = commands.add_parser("experiment", help="Rodar baseline + proposta e gravar resultados.csv")
    p.add_argument("args", nargs=argparse.REMAINDER, help="Argumentos repassados a rodar_experimento.py")
    p.set_defaults(handler=_cmd_experiment)

    p = commands.add_parser("analyze", help="Analisar resultados.csv e gerar gráficos/relatório")
    p.add_argument("args", nargs=argparse.REMAINDER, help="Argumentos repassados a analisar.py")
    p.set_defaults(handler=_cmd_analyze)

    return parser


_PASS_THROUGH = {
    "experiment": "rodar_experimento.py",
    "analyze": "analisar.py",
}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Os scripts de experimento têm seus próprios argumentos (--repeticoes,
    # --saida, ...). Repassamos tudo depois do subcomando sem reinterpretar.
    if argv and argv[0] in _PASS_THROUGH:
        return _run_script(EXPERIMENTS_DIR / _PASS_THROUGH[argv[0]], argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.handler(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
