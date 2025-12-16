import logging
from injector import inject, singleton
from llama_index.core.llms import LLM
from llama_index.llms.ollama import Ollama

from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

@singleton
class LLMComponent:
    llm: LLM

    @inject
    def __init__(self, settings: Settings) -> None:
        logger.info("Initializing Ollama LLM")
        
        self.llm = Ollama(
            model=settings.ollama.llm_model,
            base_url=settings.ollama.api_base,
            request_timeout=settings.ollama.request_timeout,
            temperature=settings.llm.temperature,
            context_window=settings.llm.context_window,
        )
