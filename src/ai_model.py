import os
import platform
import shutil
import subprocess
import requests
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

import log
from data_access import DataModel

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

class AIModelDM(DataModel):
    """Represents an AI model (local or remote)."""

    table_name = "ai_models"
    schema = {
        "name": "TEXT NOT NULL",
        "type": "TEXT NOT NULL",    # 'local' | 'remote'
        "status": "TEXT NOT NULL",  # 'up' | 'down'
        "url": "TEXT",
    }

    def __init__(
        self,
        name: str,
        type: str,
        status: str = "down",
        url: str = "",
        id: Optional[int] = None,
    ):
        self.id = id
        self.name = name
        self.type = type
        self.status = status
        self.url = url

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #
    def display(self) -> str:
        """Return a human-readable string of the model."""
        icon = "🟢" if self.status == "up" else "🔴"
        return f"{icon} {self.name} | {self.type} | {self.status}"

    def __repr__(self) -> str:
        return self.display()

    # ------------------------------------------------------------------ #
    # List
    # ------------------------------------------------------------------ #
    @classmethod
    def list(cls) -> List[Dict[str, Any]]:
        """List all AI models stored in the DB."""
        rows = cls.read(many=True)
        log.debug(f"Found {len(rows)} AI model(s).")
        return rows

    @classmethod
    def list_objects(cls) -> List["AIModelDM"]:
        """Return list of AIModelDM objects from the DB."""
        return [cls(**row) for row in cls.list()]

    # ------------------------------------------------------------------ #
    # Update status
    # ------------------------------------------------------------------ #
    @classmethod
    def update(cls, name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Check if one (or all) AI model(s) is up and update status in DB.

        Returns the updated rows.
        """
        rows = cls.read({"name": name}, many=False) if name else cls.read(many=True)
        if name and not rows:
            log.warning(f"AI model '{name}' not found.")
            return []
        if not name:
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
        """Return True if the model endpoint is reachable."""
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
    def host(cls, name: str) -> bool:
        """Host an AI model locally via Ollama."""
        log.info(f"Hosting local AI model: {name}")

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
        cls._upsert(name=name, type_="local", status="up", url=OLLAMA_HOST)
        log.success(f"AI model '{name}' hosted locally.")
        return True

    # ------------------------------------------------------------------ #
    # Stop (local)
    # ------------------------------------------------------------------ #
    @classmethod
    def stop(cls, name: Optional[str] = None) -> bool:
        """Stop (unload) a local AI model from memory.

        If `name` is None, stops all currently running local models.
        Updates the `status` to 'down' in the DB on success.
        """
        if name:
            return cls._stop_one(name)

        # Stop all running local models
        running = cls._list_running_local()
        if not running:
            log.info("No local AI models are currently running.")
            return True

        all_ok = True
        for model_name in running:
            if not cls._stop_one(model_name):
                all_ok = False
        return all_ok

    @classmethod
    def _stop_one(cls, name: str) -> bool:
        row = cls.read({"name": name}, many=False)
        if not row:
            log.warning(f"AI model '{name}' not found in DB.")
            return False

        if row["type"] != "local":
            log.warning(f"'{name}' is not a local model (type={row['type']}). "
                        "Only local models can be stopped.")
            return False

        url = row.get("url") or OLLAMA_HOST

        if not cls._is_model_running(name):
            log.info(f"Model '{name}' is not running.")
            cls._set_status(name, "down")
            return True

        log.info(f"Stopping local AI model: {name}")
        if not cls._unload_model(name, url):
            log.error(f"Failed to stop model '{name}'.")
            return False

        cls._set_status(name, "down")
        log.success(f"AI model '{name}' stopped.")
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
    def _set_status(cls, name: str, status: str) -> None:
        """Update a model's status in the DB."""
        super().update({"name": name}, {"status": status})

    # ------------------------------------------------------------------ #
    # Connect (remote)
    # ------------------------------------------------------------------ #
    @classmethod
    def connect(cls, name: str, url: str) -> bool:
        """Connect to a remote AI model."""
        log.info(f"Connecting to remote AI model: {name} @ {url}")
        if not cls._ping(name, "remote", url):
            log.error(f"Could not connect to remote AI model '{name}' at {url}.")
            return False

        cls._upsert(name=name, type_="remote", status="up", url=url)
        log.success(f"Connected to remote AI model '{name}'.")
        return True

    # ------------------------------------------------------------------ #
    # Call
    # ------------------------------------------------------------------ #
    @classmethod
    def call(cls, name: str, prompt: str) -> Optional[str]:
        """Call a specific AI model with a prompt and return the answer."""
        row = cls.read({"name": name}, many=False)
        if not row:
            log.error(f"AI model '{name}' not found.")
            return None

        url = row.get("url") or OLLAMA_HOST
        try:
            if row["type"] == "local":
                payload = {"model": name, "prompt": prompt, "stream": False}
                resp = requests.post(f"{url}/api/generate", json=payload, timeout=120)
                resp.raise_for_status()
                answer = resp.json().get("response", "")
            else:
                # Generic remote OpenAI-compatible endpoint
                payload = {
                    "model": name,
                    "messages": [{"role": "user", "content": prompt}],
                }
                resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
                resp.raise_for_status()
                answer = resp.json()["choices"][0]["message"]["content"]

            log.debug(f"Model '{name}' responded.")
            return answer
        except Exception as e:
            log.error(f"Failed to call model '{name}': {e}")
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def _upsert(cls, name: str, type_: str, status: str, url: str) -> None:
        existing = cls.read({"name": name}, many=False)
        if existing:
            super().update({"name": name}, {"type": type_, "status": status, "url": url})
            log.debug(f"Updated AI model '{name}' in DB.")
        else:
            cls.create({"name": name, "type": type_, "status": status, "url": url})
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
