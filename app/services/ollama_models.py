import json
import urllib.request


def list_chat_models(base_url: str) -> list[str]:
    with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as response:
        data = json.load(response)

    return [
        model["name"]
        for model in data.get("models", [])
        if "completion" in model.get("capabilities", [])
    ]
