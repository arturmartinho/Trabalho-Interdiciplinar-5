import os
import platform
import shutil
import subprocess
import requests
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

from . import log
from .data_access import DataModel

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Classes usadas pelo load balancer. "reserva" é o fallback.
ROLES = ("simples", "codigo", "complexo", "reserva", "geral")
_ROLE_ALIASES = {
    "código": "codigo",
    "codigo": "codigo",
    "code": "codigo",
    "simples": "simples",
    "simple": "simples",
    "complexo": "complexo",
    "complex": "complexo",
    "reserva": "reserva",
    "fallback": "reserva",
    "geral": "geral",
    "general": "geral",
}


def normalize_role(role: Optional[str]) -> str:
    """Normaliza a classe informada pelo usuário."""
    value = (role or "").strip().lower()
    return _ROLE_ALIASES.get(value, value)


class AIModelDM(DataModel):
    """Represents an AI model (local or remote)."""

    table_name = "ai_models"
    schema = {
        "name": "TEXT NOT NULL",
        "type": "TEXT NOT NULL",    # 'local' | 'remote'
        "status": "TEXT NOT NULL",  # 'up' | 'down'
        "url": "TEXT",
        "role": "TEXT",             # 'simples' | 'codigo' | 'complexo' | 'reserva' | 'geral'
    }

    def __init__(
        self,
        name: str,
        type: str,
        status: str = "down",
        url: str = "",
        role: Optional[str] = "",
        id: Optional[int] = None,
    ):
        self.id = id
        self.name = name
        self.type = type
        self.status = status
        self.url = url
        self.role = normalize_role(role)

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #
    def display(self) -> str:
        """Return a human-readable string of the model."""
        icon = "[up]" if self.status == "up" else "[down]"
        role = self.role or "-"
        return f"{self.id}. {icon} {self.name} | {self.type} | {self.status} | {role}"

    def __repr__(self) -> str:
        return self.display()

    # ------------------------------------------------------------------ #
    # List
    # ------------------------------------------------------------------ #
    @classmethod
    def list(cls) -> List[Dict[str, Any]]:
        """List all AI models stored in the DB."""
        rows = super().read(many=True)
        log.debug(f"Found {len(rows)} AI model(s).")
        return rows

    @classmethod
    def list_objects(cls) -> List["AIModelDM"]:
        """Return list of AIModelDM objects from the DB."""
        return [cls(**row) for row in cls.list()]

    @classmethod
    def get(cls, id: Optional[int] = None, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Find one model by id or name."""
        if id is not None:
            return super().read({"id": id}, many=False)
        if name:
            row = super().read({"name": name}, many=False)
            if row:
                return row
            base = name.split(":")[0]
            for candidate in super().read(many=True):
                if candidate and candidate["name"].split(":")[0] == base:
                    return candidate
        return None

    # ------------------------------------------------------------------ #
    # Register (manual / CLI)
    # ------------------------------------------------------------------ #
    @classmethod
    def register(
        cls,
        name: str,
        type_: str = "local",
        url: str = "",
        role: Optional[str] = "",
        status: str = "down",
    ) -> bool:
        """Register (or update) a model without contacting it.

        Use :meth:`host` or :meth:`connect` when the model should also be
        installed/started or validated over the network.
        """
        if not name.strip():
            log.error("Model name cannot be empty.")
            return False
        if type_ not in ("local", "remote"):
            log.error(f"Invalid model type: {type_} (use 'local' or 'remote').")
            return False
        if status not in ("up", "down"):
            log.error(f"Invalid status: {status} (use 'up' or 'down').")
            return False

        cls._upsert(
            name=name.strip(),
            type_=type_,
            status=status,
            url=url.strip(),
            role=normalize_role(role),
        )
        log.success(f"AI model '{name}' registered as {type_}/{status}.")
        return True

    # ------------------------------------------------------------------ #
    # Update status
    # ------------------------------------------------------------------ #
    @classmethod
    def update(cls, id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Check if one (or all) AI model(s) is up and update status in DB.

        Returns the updated rows.
        """
        rows = super().read({"id": id}, many=False) if id is not None else super().read(many=True)
        if id is not None and not rows:
            log.warning(f"AI model id={id} not found.")
            return []
        if id is None:
            rows = rows if isinstance(rows, list) else [rows]

        updated: List[Dict[str, Any]] = []
        for row in rows:
            if not row:
                continue
            is_up = cls._ping(row["name"], row["type"], row.get("url") or OLLAMA_HOST)
            new_status = "up" if is_up else "down"
            if new_status != row["status"]:
                super().update({"id": row["id"]}, {"status": new_status})
                row["status"] = new_status
                log.info(f"Model '{row['name']}' status -> {new_status}")
            else:
                log.debug(f"Model '{row['name']}' status unchanged ({new_status}).")
            updated.append(row)
        return updated

    @staticmethod
    def _ping(name: str, type_: str, url: str) -> bool:
        """Return True if the model endpoint is reachable.

        `name` here is the actual model name (e.g. Ollama model tag),
        not a DB identifier.
        """
        try:
            if type_ == "local":
                resp = requests.get(f"{url}/api/tags", timeout=3)
                if resp.status_code != 200:
                    return False
                tags = resp.json().get("models", [])
                return any(m.get("name", "").split(":")[0] == name.split(":")[0] for m in tags)
            else:
                # Remote: just check the URL is reachable
                resp = requests.get(url, timeout=3)
                return resp.status_code < 500
        except Exception as e:
            log.debug(f"Ping failed for '{name}': {e}")
            return False

    # ------------------------------------------------------------------ #
    # Host (local)
    # ------------------------------------------------------------------ #
    @classmethod
    def host(cls, name: str, role: Optional[str] = "") -> bool:
        """Host an AI model locally via Ollama.

        Only one local model can be 'up' at a time: any other local model
        currently running is stopped before starting this one.
        """
        log.info(f"Hosting local AI model: {name}")

        row = super().read({"name": name, "type": "local", "status": "up"}, many=False)
        if row:
            log.warning(f"Model '{name}' is already hosted")
            if role:
                super().update({"id": row["id"]}, {"role": normalize_role(role)})
            return True

        # 1. Check ollama installed
        if not cls._is_ollama_installed():
            log.warning("Ollama not found. Installing...")
            if not cls._install_ollama():
                log.error("Failed to install Ollama.")
                return False
        else:
            log.debug("Ollama is installed.")

        # 2. Check model installed
        if not cls._is_model_installed(name):
            log.warning(f"Model '{name}' not found locally. Pulling...")
            if not cls._pull_model(name):
                log.error(f"Failed to pull model '{name}'.")
                return False
        else:
            log.debug(f"Model '{name}' already installed.")

        # 3. Check model running
        if not cls._is_model_running(name):
            log.warning(f"Model '{name}' not running. Starting...")
            if not cls._run_model(name):
                log.error(f"Failed to start model '{name}'.")
                return False
        else:
            log.debug(f"Model '{name}' is running.")

        # 4. Persist
        cls._upsert(
            name=name,
            type_="local",
            status="up",
            url=OLLAMA_HOST,
            role=normalize_role(role),
        )
        log.success(f"AI model '{name}' hosted locally.")
        return True

    # ------------------------------------------------------------------ #
    # Stop (local)
    # ------------------------------------------------------------------ #
    @classmethod
    def stop(cls, id: Optional[int] = None) -> bool:
        """Stop (unload) a local AI model from memory.

        If `id` is None, stops all currently running local models.
        Updates the `status` to 'down' in the DB on success.
        """
        if id is not None:
            return cls._stop_one(id)

        # Stop all running local models
        running = cls._list_running_local()
        if not running:
            log.info("No local AI models are currently running.")
            return True

        all_ok = True
        for model_name in running:
            # Resolve the running Ollama model name back to a DB row by name
            row = super().read({"name": model_name, "type": "local"}, many=False)
            if not row:
                # Try matching by base name (without tag)
                base = model_name.split(":")[0]
                for candidate in super().read(many=True):
                    if candidate and candidate["name"].split(":")[0] == base:
                        row = candidate
                        break
            if not row:
                log.warning(f"Running model '{model_name}' has no DB entry; skipping DB update.")
                continue
            if not cls._stop_one(row["id"]):
                all_ok = False
        return all_ok

    @classmethod
    def _stop_one(cls, id: int) -> bool:
        row = super().read({"id": id}, many=False)
        if not row:
            log.warning(f"AI model id={id} not found in DB.")
            return False

        name = row["name"]

        if row["type"] != "local":
            log.warning(f"'{name}' (id={id}) is not a local model (type={row['type']}). "
                        "Only local models can be stopped.")
            return False

        url = row.get("url") or OLLAMA_HOST

        if not cls._is_model_running(name):
            log.info(f"Model '{name}' is not running.")
            cls._set_status(id, "down")
            return True

        log.info(f"Stopping local AI model: {name} (id={id})")
        if not cls._unload_model(name, url):
            log.error(f"Failed to stop model '{name}'.")
            return False

        cls._set_status(id, "down")
        log.success(f"AI model '{name}' stopped.")
        return True

    # ------------------------------------------------------------------ #
    # Resume (local)
    # ------------------------------------------------------------------ #
    @classmethod
    def resume(cls, id: Optional[int] = None) -> bool:
        """Resume (load) a local AI model into memory.

        If `id` is None, resumes all local models currently marked as 'down'
        in the DB. Updates the `status` to 'up' in the DB on success.
        """
        if id is not None:
            return cls._resume_one(id)

        candidates = super().read({"type": "local", "status": "down"}, many=True)
        if not candidates:
            log.info("No local AI models to resume.")
            return True

        all_ok = True
        for row in candidates:
            if not cls._resume_one(row["id"]):
                all_ok = False
        return all_ok

    @classmethod
    def _resume_one(cls, id: int) -> bool:
        row = super().read({"id": id}, many=False)
        if not row:
            log.warning(f"AI model id={id} not found in DB.")
            return False

        name = row["name"]

        if row["type"] != "local":
            log.warning(f"'{name}' (id={id}) is not a local model (type={row['type']}). Only local models can be resumed.")
            return False

        url = row.get("url") or OLLAMA_HOST

        if cls._is_model_running(name):
            log.info(f"Model '{name}' is already running.")
            cls._set_status(id, "up")
            return True

        # Make sure the model is installed before trying to run it
        if not cls._is_model_installed(name):
            log.warning(f"Model '{name}' not installed locally. Pulling...")
            if not cls._pull_model(name):
                log.error(f"Failed to pull model '{name}'.")
                return False

        log.info(f"Resuming local AI model: {name} (id={id})")
        if not cls._run_model(name):
            log.error(f"Failed to resume model '{name}'.")
            return False

        cls._set_status(id, "up")
        log.success(f"AI model '{name}' resumed.")
        return True

    @staticmethod
    def _unload_model(name: str, url: str) -> bool:
        """Unload a model from Ollama memory.

        Uses `keep_alive: 0`, which tells Ollama to unload the model
        immediately after the (empty) request completes.
        """
        try:
            resp = requests.post(
                f"{url}/api/generate",
                json={"model": name, "prompt": "", "keep_alive": 0},
                timeout=30,
            )
            if resp.status_code != 200:
                log.debug(f"Unload request returned {resp.status_code}: {resp.text}")
                # Fallback: try /api/chat (some versions)
                resp = requests.post(
                    f"{url}/api/chat",
                    json={
                        "model": name,
                        "messages": [],
                        "keep_alive": 0,
                    },
                    timeout=30,
                )
                if resp.status_code != 200:
                    return False
            return not AIModelDM._is_model_running(name)
        except Exception as e:
            log.error(f"Failed to unload '{name}': {e}")
            return False

    #TODO: Ollama runs only one model at a time
    @staticmethod
    def _list_running_local() -> List[str]:
        """Return names of currently running local Ollama models."""
        try:
            resp = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=5)
            if resp.status_code != 200:
                return []
            models = resp.json().get("models", [])
            return [m.get("name", "") for m in models if m.get("name")]
        except Exception as e:
            log.debug(f"Could not list running models: {e}")
            return []

    @classmethod
    def _set_status(cls, model_id: int, status: str) -> None:
        """Update a model's status in the DB by ID."""
        super().update({"id": model_id}, {"status": status})

    # ------------------------------------------------------------------ #
    # Connect (remote)
    # ------------------------------------------------------------------ #
    @classmethod
    def connect(cls, name: str, url: str, role: Optional[str] = "") -> bool:
        """Connect to a remote AI model (idempotente)."""
        log.info(f"Connecting to remote AI model: {name} @ {url}")
        if not cls._ping(name, "remote", url):
            log.error(f"Could not connect to remote AI model '{name}' at {url}.")
            return False

        cls._upsert(
            name=name,
            type_="remote",
            status="up",
            url=url,
            role=normalize_role(role),
        )
        log.success(f"Connected to remote AI model '{name}'.")
        return True

    # ------------------------------------------------------------------ #
    # Call
    # ------------------------------------------------------------------ #
    @classmethod
    def call(cls, id: int, prompt: str) -> Optional[str]:
        """Call a specific AI model with a prompt and return the answer."""
        row = super().read({"id": id}, many=False)
        if not row:
            log.error(f"AI model '{id}' not found.")
            return None

        url = row.get("url") or OLLAMA_HOST
        try:
            if row["type"] == "local":
                payload = {"model": row["name"], "prompt": prompt, "stream": False}
                resp = requests.post(f"{url}/api/generate", json=payload, timeout=120)
                resp.raise_for_status()
                answer = resp.json().get("response", "")
            else:
                # Generic remote OpenAI-compatible endpoint
                payload = {
                    "model": row["name"],
                    "messages": [{"role": "user", "content": prompt}],
                }
                resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
                resp.raise_for_status()
                answer = resp.json()["choices"][0]["message"]["content"]

            log.debug(f"Model '{id}' responded.")
            return answer
        except Exception as e:
            log.error(f"Failed to call model '{id}': {e}")
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def _upsert(
        cls,
        name: str,
        type_: str,
        status: str,
        url: str,
        role: Optional[str] = None,
    ) -> None:
        existing = super().read({"name": name, "type": type_}, many=False)
        if existing:
            updates = {"type": type_, "status": status, "url": url}
            if role:
                updates["role"] = role
            super().update({"id": existing["id"]}, updates)
            log.debug(f"Updated AI model '{name}' in DB.")
        else:
            super().create(
                {
                    "name": name,
                    "type": type_,
                    "status": status,
                    "url": url,
                    "role": role or "",
                }
            )
            log.debug(f"Inserted AI model '{name}' into DB.")

    @staticmethod
    def _is_ollama_installed() -> bool:
        return shutil.which("ollama") is not None

    @staticmethod
    def _install_ollama() -> bool:
        system = platform.system().lower()
        try:
            if system == "linux":
                subprocess.run(
                    "curl -fsSL https://ollama.com/install.sh | sh",
                    shell=True, check=True,
                )
            elif system == "darwin":
                if shutil.which("brew"):
                    subprocess.run(["brew", "install", "ollama"], check=True)
                else:
                    log.error("Homebrew not found. Install Ollama manually.")
                    return False
            elif system == "windows":
                log.error("Please install Ollama manually on Windows: https://ollama.com/download")
                return False
            else:
                log.error(f"Unsupported OS: {system}")
                return False
            return AIModelDM._is_ollama_installed()
        except subprocess.CalledProcessError as e:
            log.error(f"Ollama install failed: {e}")
            return False

    @staticmethod
    def _is_model_installed(name: str) -> bool:
        try:
            resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            base = name.split(":")[0]
            return any(m.get("name", "").split(":")[0] == base for m in models)
        except Exception:
            return False

    @staticmethod
    def _pull_model(name: str) -> bool:
        try:
            # Use ollama CLI so it works with HF-backed names too
            subprocess.run(["ollama", "pull", name], check=True)
            return True
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to pull '{name}': {e}")
            return False

    @staticmethod
    def _is_model_running(name: str) -> bool:
        try:
            resp = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=5)
            if resp.status_code != 200:
                return False
            running = resp.json().get("models", [])
            base = name.split(":")[0]
            return any(m.get("name", "").split(":")[0] == base for m in running)
        except Exception:
            return False

    @staticmethod
    def _run_model(name: str) -> bool:
        """Start the model by sending a small request (Ollama loads on demand)."""
        try:
            requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": name, "prompt": "hi", "stream": False},
                timeout=120,
            )
            return AIModelDM._is_model_running(name)
        except Exception as e:
            log.error(f"Failed to run '{name}': {e}")
            return False
