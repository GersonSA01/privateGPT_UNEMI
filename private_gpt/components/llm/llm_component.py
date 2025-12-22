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
        
        # Configurar modo JSON si está habilitado
        json_mode = getattr(settings.ollama, 'json_mode', False)
        if json_mode:
            logger.info("JSON mode enabled for Ollama LLM")
        
        self.llm = Ollama(
            model=settings.ollama.llm_model,
            base_url=settings.ollama.api_base,
            request_timeout=settings.ollama.request_timeout,
            temperature=settings.llm.temperature,
            context_window=settings.llm.context_window,
            json_mode=json_mode,
            keep_alive=settings.ollama.keep_alive,
        )
