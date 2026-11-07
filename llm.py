"""One function, ask_json(prompt), works with any provider. Swap via env vars OR per-session config:

  LLM_PROVIDER = gemini | anthropic | openai   (openai = any OpenAI-compatible API, e.g. Groq, OpenRouter)
  LLM_MODEL    = model name for that provider
  LLM_API_KEY  = key (free tiers are enough)
  LLM_BASE_URL = only for provider=openai
  LLM_FALLBACK_MODEL = optional second model, used automatically if the main one is overloaded or out of daily quota
  LLM_PROVIDER=dryrun just saves the prompt, no API call (for testing).

A web app sets CONFIG (a ContextVar) per user session instead, so two users never share a key or a cache.
Responses are cached on disk by prompt hash, so reruns are free and reproducible.
"""
import os, json, re, time, hashlib, contextvars, requests

DEFAULTS = {"gemini": "gemini-2.5-flash", "anthropic": "claude-sonnet-4-6", "openai": "gpt-4o-mini"}
CONFIG = contextvars.ContextVar("llm_config", default=None)   # {"provider","model","key","base_url","cache_dir"}


class QuotaError(RuntimeError):
    """Daily quota exhausted: retrying will not help, switch model/provider or wait for the reset."""


class TransientError(RuntimeError):
    """Retries exhausted on a temporary error (overloaded model, timeouts): a different model may still work."""


def _cfg():
    c = CONFIG.get() or {}
    provider = c.get("provider") or os.environ.get("LLM_PROVIDER", "gemini")
    return {"provider": provider,
            "model": c.get("model") or os.environ.get("LLM_MODEL") or DEFAULTS.get(provider, ""),
            "key": c.get("key") or os.environ.get("LLM_API_KEY", ""),
            "base_url": c.get("base_url") or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            "fallback": c.get("fallback_model") or os.environ.get("LLM_FALLBACK_MODEL", ""),
            "cache_dir": c.get("cache_dir") or os.environ.get("LLM_CACHE_DIR", "data/llm_cache")}


def _extract_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.S)   # first JSON-looking block
        if m:
            return json.loads(m.group(1))
        raise


def _call(cfg, prompt, system, temperature):
    provider, model, key = cfg["provider"], cfg["model"], cfg["key"]
    if provider == "gemini":
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": key},
            json={"systemInstruction": {"parts": [{"text": system}]},
                  "contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}},
            timeout=300)
        r.raise_for_status()
        data = r.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            c0 = (data.get("candidates") or [{}])[0]
            raise KeyError(f"no text in Gemini response (finishReason={c0.get('finishReason')}, promptFeedback={data.get('promptFeedback')})")
    if provider == "anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": 8000, "temperature": temperature, "system": system,
                  "messages": [{"role": "user", "content": prompt}]}, timeout=300)
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    if provider == "openai":
        r = requests.post(f"{cfg['base_url'].rstrip('/')}/chat/completions", headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "temperature": temperature, "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]},
            timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    raise ValueError(f"unknown LLM_PROVIDER {provider}")


def _attempts(cfg, prompt, system, temperature, tag, path):
    provider, model = cfg["provider"], cfg["model"]
    last = ""
    for attempt in range(5):
        try:
            raw = _call(cfg, prompt, system, temperature)
            parsed = _extract_json(raw)
            json.dump({"provider": provider, "model": model, "raw": raw, "parsed": parsed}, open(path, "w"), indent=1)
            return parsed
        except requests.HTTPError as e:
            code, body = e.response.status_code, e.response.text
            if code == 429 and re.search(r"per ?day|PerDay", body, re.I):
                raise QuotaError(f"daily quota exhausted for {provider}:{model}. Use another model or key, "
                                 f"or retry after the daily reset.") from e
            if code in (400, 401, 403, 404):      # retrying cannot fix these
                hint = {404: "model name not found: check the model name against your key's model list",
                        401: "bad API key", 403: "key not allowed for this model/API", 400: "request rejected"}[code]
                raise RuntimeError(f"HTTP {code} from {provider}:{model} - {hint}. Details: {body[:300]}") from e
            try: msg = json.loads(body)["error"]["message"]
            except Exception: msg = body
            last = f"HTTP {code}: {' '.join(msg.split())[:220]}"
            wait = 2 ** attempt * 5
            m = re.search(r"retry in ([\d.]+)s", body)             # the provider says how long to wait
            if code == 429 and m: wait = min(max(float(m.group(1)) + 1, 5), 90)
            print(f"[llm] {tag} ({model}) attempt {attempt+1}: {last[:80]}; retry in {wait:.0f}s")
            time.sleep(wait)
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            last = f"{type(e).__name__}: {' '.join(str(e).split())[:220]}"
            wait = 2 ** attempt * 5
            print(f"[llm] {tag} ({model}) attempt {attempt+1} failed ({last[:80]}); retry in {wait}s")
            time.sleep(wait)
    raise TransientError(f"LLM call failed for {tag} on {model} after 5 attempts. Last error: {last}")


def ask_json(prompt, system="You are a careful assistant. Reply with valid JSON only.", temperature=0.2, tag=""):
    cfg = _cfg()
    provider = cfg["provider"]
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    models = [cfg["model"]] + ([cfg["fallback"]] if cfg["fallback"] and cfg["fallback"] != cfg["model"] else [])
    path_for = lambda m: os.path.join(cfg["cache_dir"], f"{tag}_{hashlib.sha256(f'{provider}|{m}|{system}|{prompt}'.encode()).hexdigest()[:16]}.json")
    for m in models:                                          # a cached answer from any configured model wins
        if os.path.exists(path_for(m)):
            return json.load(open(path_for(m)))["parsed"]
    if provider == "dryrun":
        open(os.path.join(cfg["cache_dir"], f"DRYRUN_{tag}.txt"), "w").write(prompt)
        print(f"[dryrun] {tag}: prompt ~{len(prompt)//4} tokens saved")
        return None
    if not cfg["key"]:
        raise RuntimeError("No API key set (LLM_API_KEY).")
    err = None
    for i, m in enumerate(models):
        try:
            return _attempts(dict(cfg, model=m), prompt, system, temperature, tag, path_for(m))
        except (QuotaError, TransientError) as e:
            err = e
            if i + 1 < len(models):
                print(f"[llm] {tag}: {m} unavailable; switching to fallback model {models[i+1]}")
    raise err
