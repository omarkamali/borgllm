import json
import sys
from urllib import request, error
from urllib.parse import urljoin
from typing import Dict, List

from borgllm.borgllm import BorgLLM, BUILTIN_PROVIDERS


def _fetch_models(base_url: str, api_key: str) -> List[str]:
    endpoint = urljoin(base_url.rstrip("/") + "/", "models")
    req = request.Request(endpoint)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as e:
        try:
            details = e.read().decode("utf-8")
        except Exception:
            details = ""
        raise RuntimeError(f"HTTP {e.code} {e.reason} {details}") from None
    except Exception as e:
        raise RuntimeError(str(e)) from None

    out: List[str] = []
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            for item in payload["data"]:
                mid = item.get("id") or item.get("name")
                if isinstance(mid, str):
                    out.append(mid)
        elif isinstance(payload.get("models"), list):
            for item in payload["models"]:
                mid = item.get("id") or item.get("name")
                if isinstance(mid, str):
                    out.append(mid)
    return sorted(set(out))


def list_all_providers_models() -> Dict[str, List[str]]:
    llm = BorgLLM.get_instance()
    results: Dict[str, List[str]] = {}

    for provider_name in BUILTIN_PROVIDERS.keys():
        settings = BUILTIN_PROVIDERS[provider_name]
        model_ref = settings.get("default_model")
        try:
            cfg = llm.get(f"{provider_name}:{model_ref}")
        except Exception:
            continue
        try:
            models = _fetch_models(str(cfg.base_url), cfg.api_key)
            results[provider_name] = models
        except Exception:
            results[provider_name] = []
    return results


def main() -> int:
    data = list_all_providers_models()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
