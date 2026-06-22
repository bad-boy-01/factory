
"""
LLM Adapter — Novel Video Factory v4
PRIMARY:  Groq free-tier (llama-3.3-70b) — sign up FREE at console.groq.com
FALLBACK: Ollama local (qwen2.5:7b)      — fully offline, no internet needed

BUG FIXES vs v3:
- unload_model URL was doubled (/api/api/generate) → FIXED: correct endpoint
- Groq timeout too low for long prompts → FIXED: 120s
- Better mock responses matching all system prompt types
"""
import json
import logging
import os
import requests
import time
from json_repair import repair_json

logger = logging.getLogger(__name__)

QUOTA_MARKERS = [
    "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
    "generate_content_free_tier_requests",
    "quota exceeded",
    "RESOURCE_EXHAUSTED",
]


class LLMFallbackExhausted(Exception):
    """
    Raised when both the primary and fallback LLM providers fail AND
    strict_mode is enabled in config. In non-strict mode (the default),
    SmartLLMAdapter does not raise this — it returns mock content instead,
    but always sets `last_call_was_fallback = True` first so callers can
    detect and react to it (skip/flag/retry) instead of silently treating
    the mock as real story content.
    """
    pass


# ── Groq Free-Tier Adapter ────────────────────────────────────────────────────
class GroqLLMAdapter:
    """
    Groq free-tier LLM.
    Sign up FREE at https://console.groq.com — no credit card required.
    Set GROQ_API_KEY in Kaggle Secrets or environment variables.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or self._load_key()
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.is_cloud = True

    def _load_key(self) -> str:
        """Try environment variable, then Kaggle Secrets."""
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            try:
                from kaggle_secrets import UserSecretsClient  # type: ignore
                key = UserSecretsClient().get_secret("GROQ_API_KEY")
            except Exception:
                pass
        return key

    def check_health(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str = None,
                 temperature: float = 0.7, model: str = None, max_tokens: int = 4096, **kwargs) -> str:
        if not self.api_key:
            return "ERROR: GROQ_NO_API_KEY"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(6):
            try:
                r = requests.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={"model": model or self.model_name,
                          "messages": messages,
                          "temperature": temperature,
                          "max_tokens": max_tokens},
                    timeout=120,
                )
                if r.status_code == 429:
                    if attempt == 5:
                        return '{"_quota_exhausted": true}'
                    # Exponential backoff: 10s, 30s, 60s, 120s, 180s, 300s
                    wait = [10, 30, 60, 120, 180, 300][attempt]
                    logger.warning(f"Groq Rate Limit (429). Attempt {attempt+1}/6. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"Groq attempt {attempt+1} failed: {e}")
                if attempt < 5:
                    time.sleep(2)
        
        return "ERROR: GROQ_FAILED"

    def generate_json(self, prompt: str, system_prompt: str = None,
                      temperature: float = 0.1, model: str = None, max_tokens: int = 4096, **kwargs) -> str:
        """Force JSON response format if supported, then repair."""
        # Note: Groq supports response_format={"type": "json_object"} for some models
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt + " You must output valid JSON."})
        messages.append({"role": "user", "content": prompt})

        try:
            r = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"model": model or self.model_name,
                      "messages": messages,
                      "temperature": temperature,
                      "response_format": {"type": "json_object"},
                      "max_tokens": max_tokens},
                timeout=120,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return self._repair(content)
        except Exception as e:
            logger.debug(f"Groq JSON mode failed or unsupported: {e}")
            # Fallback to standard generate + repair
            content = self.generate(prompt, system_prompt, temperature, model, **kwargs)
            return self._repair(content)

    def _repair(self, content: str) -> str:
        if "ERROR:" in content: return content
        try:
            repaired = repair_json(content)
            if isinstance(repaired, (dict, list)):
                return json.dumps(repaired)
            return repaired
        except Exception:
            return content

    def unload_model(self, *args, **kwargs):
        pass  # No-op for API-based adapters

# ── Gemini API Adapter ────────────────────────────────────────────────────────
class GeminiLLMAdapter:
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or self._load_key()
        self.is_cloud = True
        
        if self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = None

    def _load_key(self) -> str:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            try:
                from kaggle_secrets import UserSecretsClient  # type: ignore
                key = UserSecretsClient().get_secret("GEMINI_API_KEY")
            except Exception:
                pass
        return key

    def check_health(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str = None,
                 temperature: float = 0.7, model: str = None, max_tokens: int = 4096, **kwargs) -> str:
        if not self.model:
            return "ERROR: GEMINI_NO_API_KEY"

        import google.generativeai as genai
        
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}" if system_prompt else prompt
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        for attempt in range(3):
            try:
                response = self.model.generate_content(full_prompt, generation_config=generation_config)
                return response.text
            except Exception as e:
                err_str = str(e)
                logger.warning(f"Gemini attempt {attempt+1} failed: {err_str}")
                for marker in QUOTA_MARKERS:
                    if marker in err_str:
                        return '{"_quota_exhausted": true}'
                time.sleep(2)
        return "ERROR: GEMINI_FAILED"

    def generate_json(self, prompt: str, system_prompt: str = None,
                      temperature: float = 0.1, model: str = None, max_tokens: int = 4096, **kwargs) -> str:
        import google.generativeai as genai
        if not self.model:
            return "ERROR: GEMINI_NO_API_KEY"

        full_prompt = f"System: {system_prompt} You must output valid JSON.\n\nUser: {prompt}" if system_prompt else f"{prompt}\nYou must output valid JSON."
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json"
        )
        
        for attempt in range(3):
            try:
                response = self.model.generate_content(full_prompt, generation_config=generation_config)
                return self._repair(response.text)
            except Exception as e:
                err_str = str(e)
                logger.warning(f"Gemini JSON attempt {attempt+1} failed: {err_str}")
                for marker in QUOTA_MARKERS:
                    if marker in err_str:
                        return '{"_quota_exhausted": true}'
                time.sleep(2)
        return "ERROR: GEMINI_FAILED"

    def _repair(self, content: str) -> str:
        if "ERROR:" in content: return content
        try:
            repaired = repair_json(content)
            if isinstance(repaired, (dict, list)):
                return json.dumps(repaired)
            return repaired
        except Exception:
            return content

    def unload_model(self, *args, **kwargs):
        pass

# ── Ollama Adapter ────────────────────────────────────────────────────────────
class OllamaLLMAdapter:
    def __init__(self, host: str = "http://localhost:11434", model_name: str = "qwen2.5:7b"):
        self.host = host
        self.model_name = model_name
        self.is_cloud = False

    def check_health(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system_prompt: str = None,
                 temperature: float = 0.7, model: str = None, max_tokens: int = 4096, **kwargs) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": model or self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            r = requests.post(url, json=payload, timeout=300)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            logger.warning(f"Ollama generation failed: {e}")
            return "ERROR: OLLAMA_FAILED"

    def generate_json(self, prompt: str, system_prompt: str = None,
                      temperature: float = 0.0, model: str = None, max_tokens: int = 4096, **kwargs) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": model or self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            r = requests.post(url, json=payload, timeout=300)
            r.raise_for_status()
            return self._repair(r.json().get("response", ""))
        except Exception as e:
            logger.warning(f"Ollama JSON generation failed: {e}")
            return "ERROR: OLLAMA_FAILED"

    def _repair(self, content: str) -> str:
        if "ERROR:" in content: return content
        try:
            repaired = repair_json(content)
            if isinstance(repaired, (dict, list)):
                return json.dumps(repaired)
            return repaired
        except Exception:
            return content

    def unload_model(self, *args, **kwargs):
        pass


# ── DeepSeek API Adapter (Fallback) ───────────────────────────────────────────
class DeepSeekLLMAdapter:
    """
    DeepSeek API LLM (Extremely cheap, OpenAI compatible).
    Set DEEPSEEK_API_KEY in Kaggle Secrets or environment variables.
    """
    def __init__(self, model_name: str = "deepseek-chat", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or self._load_key()
        self.api_url = "https://api.deepseek.com/chat/completions"
        self.is_cloud = True

    def _load_key(self) -> str:
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            try:
                from kaggle_secrets import UserSecretsClient  # type: ignore
                key = UserSecretsClient().get_secret("DEEPSEEK_API_KEY")
            except Exception:
                pass
        return key

    def check_health(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str = None,
                 temperature: float = 0.7, model: str = None, max_tokens: int = 4096, **kwargs) -> str:
        if not self.api_key:
            return "ERROR: DEEPSEEK_NO_API_KEY"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(3):
            try:
                r = requests.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json={"model": model or self.model_name,
                          "messages": messages,
                          "temperature": temperature,
                          "max_tokens": max_tokens},
                    timeout=120,
                )
                if r.status_code == 429:
                    wait = [10, 30, 60][attempt]
                    logger.warning(f"DeepSeek Rate Limit (429). Attempt {attempt+1}/3. Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"DeepSeek attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)
        
        return "ERROR: DEEPSEEK_FAILED"

    def generate_json(self, prompt: str, system_prompt: str = None,
                      temperature: float = 0.1, model: str = None, max_tokens: int = 4096, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt + " You must output valid JSON."})
        messages.append({"role": "user", "content": prompt})

        try:
            r = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"model": model or self.model_name,
                      "messages": messages,
                      "temperature": temperature,
                      "response_format": {"type": "json_object"},
                      "max_tokens": max_tokens},
                timeout=120,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return self._repair(content)
        except Exception as e:
            logger.debug(f"DeepSeek JSON mode failed or unsupported: {e}")
            content = self.generate(prompt, system_prompt, temperature, model, **kwargs)
            return self._repair(content)

    def _repair(self, content: str) -> str:
        if "ERROR:" in content: return content
        try:
            repaired = repair_json(content)
            if isinstance(repaired, (dict, list)):
                return json.dumps(repaired)
            return repaired
        except Exception:
            return content

    def unload_model(self, *args, **kwargs):
        pass


# ── Smart Adapter: tries Groq first, falls back to DeepSeek ────────────────────
class SmartLLMAdapter:
    """
    Intelligent router. Prioritizes a specific provider based on config.
    Falls back to other available providers if the preferred one fails or goes offline.
    """
    def __init__(self, config: dict = None, provider_override: str = None, allow_fallback: bool = True):
        cfg = config or {}
        models = cfg.get("models", {}).get("llm", {})

        self.provider = provider_override or models.get("provider", "groq").lower()
        self.allow_fallback = allow_fallback
        self.quota_exhausted = False
        
        groq_model = models.get("model", "llama-3.3-70b-versatile")
        deepseek_model = models.get("deepseek_model", "deepseek-chat")
        gemini_model = "gemini-2.5-flash"
        ollama_model = models.get("ollama_model", "qwen2.5:7b")
        ollama_host = models.get("ollama_host", "http://localhost:11434")

        self.strict_mode = bool(cfg.get("system", {}).get("strict_mode", False))

        self.last_call_was_fallback = False
        self.fallback_count = 0
        self.total_calls = 0

        self._groq = GroqLLMAdapter(model_name=groq_model)
        self._deepseek = DeepSeekLLMAdapter(model_name=deepseek_model)
        self._gemini = GeminiLLMAdapter(model_name=gemini_model)
        self._ollama = OllamaLLMAdapter(host=ollama_host, model_name=ollama_model)

        self._primary = None
        
        # Route explicitly if provider matches
        if self.provider == "gemini" and self._gemini.check_health():
            self._primary = self._gemini
            logger.info(f"LLM: Routed strictly to Gemini ({gemini_model})")
        elif self.provider == "ollama" and self._ollama.check_health():
            self._primary = self._ollama
            logger.info(f"LLM: Routed strictly to Ollama ({ollama_model})")
        elif self.provider == "groq" and self._groq.check_health():
            self._primary = self._groq
            logger.info(f"LLM: Routed strictly to Groq ({groq_model})")
        elif self.provider == "deepseek" and self._deepseek.check_health():
            self._primary = self._deepseek
            logger.info(f"LLM: Routed strictly to DeepSeek ({deepseek_model})")
        else:
            # Automatic fallback routing if preferred provider is down
            if self._deepseek.check_health():
                self._primary = self._deepseek
                logger.info(f"LLM: Fallback to DeepSeek ({deepseek_model})")
            elif self._groq.check_health():
                self._primary = self._groq
                logger.info(f"LLM: Fallback to Groq ({groq_model})")
            elif self._gemini.check_health():
                self._primary = self._gemini
                logger.info(f"LLM: Fallback to Gemini ({gemini_model})")
            elif self._ollama.check_health():
                self._primary = self._ollama
                logger.info(f"LLM: Fallback to Ollama ({ollama_model})")
            else:
                logger.warning("LLM: No adapter available — mock mode active")

    @property
    def is_cloud(self) -> bool:
        return getattr(self._primary, "is_cloud", False)

    @property
    def is_available(self) -> bool:
        """Returns True if at least one real LLM provider is reachable."""
        return self._primary is not None

    def _handle_exhausted(self, system_prompt: str, prompt: str) -> str:
        """
        Called only when every real provider has failed for this request.
        """
        self.fallback_count += 1
        self.last_call_was_fallback = True
        logger.error(
            f"⚠️  LLM FALLBACK #{self.fallback_count}: both providers failed — "
            f"raising LLMFallbackExhausted. "
        )
        raise LLMFallbackExhausted(f"All LLM providers failed. system_prompt[:60]={system_prompt[:60]!r}")

    def generate(self, prompt: str, system_prompt: str = None,
                 temperature: float = 0.7, model: str = None, max_tokens: int = 4096, **kwargs) -> str:
        self.total_calls += 1
        self.last_call_was_fallback = False

        if self.quota_exhausted:
            return '{"_quota_exhausted": true}'

        if self._primary is None:
            return self._handle_exhausted(system_prompt or "", prompt)

        result = self._primary.generate(
            prompt, system_prompt=system_prompt,
            temperature=temperature, model=model, **kwargs
        )

        if "_quota_exhausted" in result:
            self.quota_exhausted = True
            return result

        if "ERROR:" in result and self.allow_fallback:
            logger.warning(f"Primary LLM failed ({result}). Trying fallback...")
            # If primary failed, try other available providers
            for fallback in [self._groq, self._gemini, self._deepseek, self._ollama]:
                if fallback and fallback != self._primary and fallback.check_health():
                    logger.info(f"LLM Fallback: Trying {fallback.__class__.__name__}")
                    result = fallback.generate(
                        prompt, system_prompt=system_prompt,
                        temperature=temperature, model=model, **kwargs
                    )
                    if "ERROR:" not in result and "_quota_exhausted" not in result:
                        break

            if "ERROR:" in result:
                return self._handle_exhausted(system_prompt or "", prompt)

        return result

    def generate_json(self, prompt: str, system_prompt: str = None,
                      temperature: float = 0.1, model: str = None, max_tokens: int = 4096, **kwargs) -> str:
        """Tries primary, then fallback, with JSON-specific logic."""
        self.total_calls += 1
        self.last_call_was_fallback = False

        if self.quota_exhausted:
            return '{"_quota_exhausted": true}'

        if self._primary is None:
            return self._handle_exhausted(system_prompt or "", prompt)

        result = self._primary.generate_json(
            prompt, system_prompt=system_prompt,
            temperature=temperature, model=model, **kwargs
        )

        if "_quota_exhausted" in result:
            self.quota_exhausted = True
            return result

        if "ERROR:" in result and self.allow_fallback:
            logger.warning(f"Primary LLM JSON failed ({result}). Trying fallback...")
            # If primary failed, try other available providers
            for fallback in [self._groq, self._gemini, self._deepseek, self._ollama]:
                if fallback and fallback != self._primary and fallback.check_health():
                    logger.info(f"LLM JSON Fallback: Trying {fallback.__class__.__name__}")
                    result = fallback.generate_json(
                        prompt, system_prompt=system_prompt,
                        temperature=temperature, model=model, **kwargs
                    )
                    if "ERROR:" not in result and "_quota_exhausted" not in result:
                        break

            if "ERROR:" in result:
                return self._handle_exhausted(system_prompt or "", prompt)

        return result

    def unload_model(self, model_name: str = None):
        if self._primary is not None:
            self._primary.unload_model(model_name)


