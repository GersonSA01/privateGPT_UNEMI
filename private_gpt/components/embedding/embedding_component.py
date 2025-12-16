import logging
from injector import inject, singleton
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

@singleton
class EmbeddingComponent:
    embedding_model: BaseEmbedding

    @inject
    def __init__(self, settings: Settings) -> None:
        logger.info("Initializing Ollama Embedding")
        
        self.embedding_model = OllamaEmbedding(
            model_name=settings.ollama.embedding_model,
            base_url=settings.ollama.embedding_api_base,
        )
