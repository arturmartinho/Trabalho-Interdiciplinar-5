import sys

import log
from ai_model import AIModelDM

def _pause() -> None:
    input("\nPress Enter to continue...")

def _print_models(models) -> None:
    if not models:
        print("  (no AI models registered)")
        return
    for m in models:
        obj = AIModelDM(**m) if isinstance(m, dict) else m
        print(f"  - {obj.display()}")

def ai_models_menu() -> None:
    while True:
        print("\n=== AI Models ===")
        print("  1. List AI Models")
        print("  2. Host a local AI Model")
        print("  3. Connect to a remote AI Model")
        print("  4. Update AI Model status")
        print("  5. Call an AI Model")
        print("  6. Delete an AI Model")
        print("  7. Stop a local AI Model")
        print("  0. Back")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            _print_models(AIModelDM.list())
            _pause()

        elif choice == "2":
            name = input("Model name to host (e.g. llama3.2): ").strip()
            if not name:
                print("Cancelled.")
            else:
                AIModelDM.host(name)
            _pause()

        elif choice == "3":
            name = input("Remote model name: ").strip()
            url = input("Remote URL (e.g. http://host:11434): ").strip()
            if not name or not url:
                print("Cancelled.")
            else:
                AIModelDM.connect(name, url)
            _pause()

        elif choice == "4":
            _print_models(AIModelDM.list())
            id = input("Model ID (blank = all): ").strip()
            updated = AIModelDM.update(int(id) if id else None)
            _print_models(updated)
            _pause()

        elif choice == "5":
            name = input("Model name: ").strip()
            prompt = input("Prompt: ").strip()
            if not name or not prompt:
                print("Cancelled.")
            else:
                answer = AIModelDM.call(name, prompt)
                print("\n--- Answer ---")
                print(answer if answer else "(no answer)")
            _pause()

        elif choice == "6":
            _print_models(AIModelDM.list())
            id = input("Model ID to delete: ").strip()
            if id:
                AIModelDM.delete({"id": int(id)})
                log.success(f"Deleted Model with ID = {id}.")
            _pause()

        elif choice == "7":
            _print_models(AIModelDM.list())
            id = input("Model ID to stop (blank = all running): ").strip()
            AIModelDM.stop(int(id) if id else None)
            _pause()

        elif choice == "0":
            return

        else:
            log.warning("Invalid option.")

def menu() -> None:
    while True:
        print("\n=== AI LoadBalancer ===")
        print("  1. AI Models")
        print("  0. Exit")

        choice = input("Choose an option: ").strip()

        if choice == "1":
            ai_models_menu()
        elif choice == "0":
            log.info("Goodbye!")
            sys.exit(0)
        else:
            log.warning("Invalid option.")
