import time
import random 
from typing import Optional, Dict, Any, Callable, Union
from functools import wraps
from pydantic import BaseModel
import asyncio

from google import genai
from google.genai import types

from ..utils.logging import get_logger

logger = get_logger(__name__)

def _default_is_transient(exc: Exception) -> bool:
    """
    Heuristics to decide if an exception is transient.
    Inspect common attributes (status code, code, retry_after) first,
    then fallback to string matching for 429/5xx
    """
    # check structured attributes first 
    for attr in ("status_code", "code", "http_status", "status"):
        code = getattr(exc, attr, None)
        if code is not None:
            try:
                code_int = int(code)
                if code_int == 429 or 500 <= code_int < 600:
                    return True
            except Exception:
                pass

    # check for retry-after header-like attribute
    if hasattr(exc, "retry_after"):
        return True

    # fallback to string matching 
    s = str(exc).lower()
    if "429" in s or "rate limit" in s or "quota" in s:
        return True
    if "500" in s or "internal server error" in s or "service unavailable" in s:
        return True
    
    return False

# --- 1. Exponential backoff implementation ---
def retry_with_backoff(
    max_retries: int = 5, 
    initial_delay: float = 1.0, 
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
    is_transient: Callable[[Exception], bool] = _default_is_transient):
    """
    Decorator to retry a function call with exponential backoff
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as ex:
                    attempt += 1
                    transient = is_transient(ex)
                    # if not transient or we've exhausted retries -> re-raise
                    if not transient or attempt > max_retries:
                        print("Non-retryable error or max retries reached")
                        raise

                    # respect Retry-After if available (attribute or in message)
                    retry_after = getattr(ex, "retry_after", None)
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except Exception:
                            wait = min(delay, max_delay)
                    else:
                        base = min(delay, max_delay)
                        wait = random.uniform(0, base)  # full jitter

                    print(f"""Transient error detected. Attempt {attempt}/{max_retries}.
                          Retrying in {wait:.2f}s. Error: {str(ex)}""")
                    time.sleep(wait)

                    # exponential increase in next round
                    delay = min(delay * backoff_factor, max_delay)

        return wrapper
    return decorator

# --- Async exponential backoff implementation ---
def async_retry_with_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0,
    is_transient: Callable[[Exception], bool] = _default_is_transient
):
    """
    Decorator to retry an async function call with exponential backoff
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            delay = initial_delay

            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as ex:
                    attempt += 1
                    transient = is_transient(ex)
                    # if not transient or we've exhausted retries -> re-raise
                    if not transient or attempt > max_retries:
                        print("Non-retryable error or max retries reached")
                        raise

                    # respect Retry-After if available 
                    retry_after = getattr(ex, "retry_after", None)
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except Exception:
                            wait = min(delay, max_delay)
                    else:
                        base = min(delay, max_delay)
                        wait = random.uniform(0, base)

                    print(f"""Transient error detected. Attempt {attempt}/{max_retries}.
                          Retrying in {wait:.2f}s. Error: {str(ex)}""")
                    await asyncio.sleep(wait)

                    # exponential increase in next round
                    delay = min(delay * backoff_factor, max_delay)

        return wrapper
    return decorator

class GeminiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        system_instruction: Optional[str] = None,
    ):
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()

        self.model_name = model_name
        self.system_instruction = system_instruction
        # initialize async client lazily
        self._aclient = None

    @property
    def aclient(self):
        """Lazy initialization of async client"""
        if self._aclient is None:
            self._aclient = self.client.aio
        return self._aclient
    
    @retry_with_backoff(max_retries=3)
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequencyPenalty: Optional[float] = None,
        thinking_mode: bool = False,
        response_schema: Optional[BaseModel] = None,
        **sdk_kwargs,
    ) -> Union[str, BaseModel]:
        """
        Synchronous generate wrapper
        """
        # Input validation
        if temperature is not None and not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        if top_p is not None and not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be values between 0 and 1 (both inclusive)")
        if frequencyPenalty is not None and not (-2.0 <= frequencyPenalty <= 2.0):
            raise ValueError("frequencyPenalty must be within the range between -2 and 2")
        
        config_args: Dict[str, Any] = {}
        if temperature is not None:
            config_args["temperature"] = temperature
        if top_p is not None:
            config_args["top_p"] = top_p
        if frequencyPenalty is not None:
            config_args["frequencyPenalty"] = frequencyPenalty
        if self.system_instruction:
            config_args["system_instruction"] = self.system_instruction

        if thinking_mode:
            config_args["thinking_config"] = types.ThinkingConfig(include_thoughts=True)

        # structured output hints
        if response_schema:
            config_args["response_mime_type"] = "application/json"
            config_args["response_json_schema"] = response_schema.model_json_schema()

        gen_config = types.GenerateContentConfig(**config_args)

        print(f"Sending request to model={model or self.model_name}")

        response = self.client.models.generate_content(
            model=model or self.model_name,
            contents=prompt,
            config=gen_config,
            **sdk_kwargs
        )

        if response is None:
            logger.error("Received None from LLM")
            raise ValueError("LLM API responds with None")

        # handle response
        if response_schema is not None:
            try:
                return response_schema.model_validate_json(response.text)
            except:
                return response.text
            
        return response.text
    
    @async_retry_with_backoff(max_retries=3)
    async def agenerate(
        self, 
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequencyPenalty: Optional[float] = None,
        thinking_mode: bool = False,
        response_schema: Optional[BaseModel] = None,
        **sdk_kwargs,
    ) -> Union[str, BaseModel]:
        """
        Asynchronous generate wrapper
        """
        # Input validation
        if temperature is not None and not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        if top_p is not None and not (0.0 <= top_p <= 1.0):
            raise ValueError("top_p must be values between 0 and 1 (both inclusive)")
        if frequencyPenalty is not None and not (-2.0 <= frequencyPenalty <= 2.0):
            raise ValueError("frequencyPenalty must be within the range between -2 and 2")
        
        config_args: Dict[str, Any] = {}
        if temperature is not None:
            config_args["temperature"] = temperature
        if top_p is not None:
            config_args["top_p"] = top_p
        if frequencyPenalty is not None:
            config_args["frequencyPenalty"] = frequencyPenalty
        if self.system_instruction:
            config_args["system_instruction"] = self.system_instruction

        if thinking_mode:
            config_args["thinking_config"] = types.ThinkingConfig(include_thoughts=True)

        # structured output hints
        if response_schema:
            config_args["response_mime_type"] = "application/json"
            config_args["response_json_schema"] = response_schema.model_json_schema()

        gen_config = types.GenerateContentConfig(**config_args)

        print(f"Sending async request to model={model or self.model_name}")

        response = await self.aclient.models.generate_content(
            model=model or self.model_name,
            contents=prompt,
            config=gen_config,
            **sdk_kwargs
        )

        if response is None:
            logger.error("Received None from LLM")
            raise ValueError("LLM API responds with None")

        # handle response
        if response_schema is not None:
            try:
                return response_schema.model_validate_json(response.text)
            except:
                return response.text
            
        return response.text
    
    def close(self):
        self.client.close()
        logger.info("Gemini client closed")

    async def aclose(self):
        """Close async client"""
        if self._aclient is not None:
            await self._aclient.aclose()
            logger.info("Gemini async client closed")
    
class movie_review(BaseModel):
    movie_title: str
    sentiment: str
    score: int
    key_themes: list[str]
    
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    api_key = os.environ['GEMINI_API_KEY']

    if not api_key:
        raise ValueError("system environment variable for GEMINI API not available")
    
    bot = GeminiClient(
        api_key=api_key,
        model_name="gemini-2.5-flash",
        system_instruction="You are a helpful and precise AI assistant"
    )

    # try:
    #     print("\n--- standard thinking response ---")
    #     response = bot.generate(
    #         prompt="Explain the logic behind Monty Hall Problem",
    #         thinking_mode=True,
    #         temperature=0.5,
    #         top_p=0.95,
    #         # frequencyPenalty=0.2
    #     )
    #     print(response)
    # except Exception as ex:
    #     print("Generation failed:" + str(ex))

    try:
        print("\n--- JSON output ---")
        response = bot.generate(
            prompt="Review the movie 'Avatar' briefly",
            response_schema=movie_review,
            # thinking_mode=True,
            temperature=0,
            top_p=0.9,
            # frequencyPenalty=0.2
        )
        print(response)
        print(type(response))
    except Exception as ex:
        print("Generation failed:" + str(ex))
