import logging
import tempfile
import json
import time  # <--- IMPORTANTE: Necesario para el sleep
from pathlib import Path
from typing import TYPE_CHECKING, AnyStr, BinaryIO
from qdrant_client.http import models as rest
from injector import inject, singleton
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage import StorageContext
from llama_index.core.schema import MetadataMode

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import get_ingestion_component
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.settings.settings import settings

if TYPE_CHECKING:
    from llama_index.core.storage.docstore.types import RefDocInfo

logger = logging.getLogger(__name__)


@singleton
class IngestService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.llm_service = llm_component
        self.embedding_model = embedding_component.embedding_model

        # =====================================================================
        # SOLUCIÓN 1: BATCH SIZE MÍNIMO
        # =====================================================================
        # Usamos object.__setattr__ para saltarnos la validación de Pydantic
        # si fuera necesario, aunque para atributos simples suele dejar.
        try:
            self.embedding_model.embed_batch_size = 1
        except Exception:
            object.__setattr__(self.embedding_model, "embed_batch_size", 1)
        
        # =====================================================================
        # SOLUCIÓN 2: MONKEY PATCH (ROBUSTEZ + ENFRIAMIENTO)
        # =====================================================================
        
        # 1. Capturamos el método original (bound method)
        original_embed_batch = self.embedding_model.get_text_embedding_batch

        # 2. Definimos el wrapper
        def robust_embed_batch_wrapper(texts, **kwargs):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # ENFRIAMIENTO: Pausa obligatoria para no saturar Ollama
                    time.sleep(0.5) 
                    
                    # Llamada al original
                    return original_embed_batch(texts, **kwargs)
                except Exception as e:
                    logger.warning(f"⚠️ Error embedding (Intento {attempt+1}/{max_retries}): {str(e)}")
                    # BACKOFF: Si falla, esperamos más tiempo (2s, 4s, 6s...)
                    time.sleep(2 * (attempt + 1))
                    
                    if attempt == max_retries - 1:
                        logger.error("❌ Fallaron todos los reintentos de embedding. Posible caída de Ollama.")
                        raise e

        # 3. APLICAMOS EL PARCHE (LA CORRECCIÓN CRÍTICA)
        # Usamos object.__setattr__ para "inyectar" la función a la fuerza,
        # saltándonos la protección de Pydantic que causaba el ValueError.
        object.__setattr__(
            self.embedding_model, 
            "get_text_embedding_batch", 
            robust_embed_batch_wrapper
        )
        
        logger.info("🛡️ Robust Embedding Wrapper instalado correctamente (Pydantic Bypassed).")
        # =====================================================================

        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        
        try:
            chunk_sz = settings().ingestion.chunk_size
            chunk_ov = settings().ingestion.chunk_overlap
        except AttributeError:
            # =================================================================
            # SOLUCIÓN 3: REDUCIR TAMAÑO DE CHUNK
            # =================================================================
            chunk_sz = 256  # <--- BAJADO DE 512
            chunk_ov = 30   # <--- BAJADO DE 50

        logger.info(f"⚙️ FINAL CHUNK CONFIG: Size={chunk_sz}, Overlap={chunk_ov}")

        node_parser = SentenceSplitter(
            chunk_size=chunk_sz, 
            chunk_overlap=chunk_ov
        )

        self.ingest_component = get_ingestion_component(
            self.storage_context,
            embed_model=embedding_component.embedding_model,
            transformations=[node_parser, embedding_component.embedding_model],
            settings=settings(),
        )

    def _ingest_data(self, file_name: str, file_data: AnyStr, db_id: int | None = None) -> list[IngestedDoc]:
        logger.debug("Got file data of size=%s to ingest", len(file_data))
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                path_to_tmp = Path(tmp.name)
                if isinstance(file_data, bytes):
                    path_to_tmp.write_bytes(file_data)
                else:
                    path_to_tmp.write_text(str(file_data))
                
                initial_metadata = {"db_id": db_id} if db_id else {}
                
                return self.ingest_file(file_name, path_to_tmp, metadata=initial_metadata)
            finally:
                tmp.close()
                path_to_tmp.unlink()

    def ingest_file(self, file_name: str, file_data: Path, metadata: dict = None) -> list[IngestedDoc]:
        logger.info("Ingesting file_name=%s with metadata=%s", file_name, metadata)
        documents = self.ingest_component.ingest(file_name, file_data, metadata=metadata)
        logger.info("Finished ingestion file_name=%s", file_name)
        return [IngestedDoc.from_document(document) for document in documents]

    def ingest_text(self, file_name: str, text: str) -> list[IngestedDoc]:
        logger.debug("Ingesting text data with file_name=%s", file_name)
        return self._ingest_data(file_name, text)

    def ingest_bin_data(
        self, file_name: str, raw_file_data: BinaryIO, db_id: int | None = None
    ) -> list[IngestedDoc]:
        logger.debug("Ingesting binary data with file_name=%s", file_name)
        file_data = raw_file_data.read()
        return self._ingest_data(file_name, file_data, db_id=db_id)

    def bulk_ingest(self, files: list[tuple[str, Path]], role_tag: list[str] | str = "general") -> list[IngestedDoc]:
        if isinstance(role_tag, str):
            roles_to_save = [role_tag]
        else:
            roles_to_save = role_tag
        
        logger.info("Ingesting file_names=%s with roles=%s", [f[0] for f in files], roles_to_save)
        
        documents = []
        for file_name, file_path in files:
            docs = self.ingest_file(file_name, file_path, metadata={"role": roles_to_save})
            documents.extend(docs)
            
        logger.info("Finished ingestion file_name=%s", [f[0] for f in files])
        return documents

    def list_ingested(self) -> list[IngestedDoc]:
        ingested_docs: list[IngestedDoc] = []
        
        LOCAL_DOCS_PATH = Path("source_documents") 

        try:
            docstore = self.storage_context.docstore
            ref_docs: dict[str, RefDocInfo] | None = docstore.get_all_ref_doc_info()

            if not ref_docs:
                return ingested_docs

            kv_store = getattr(docstore, "_kvstore", None)
            namespace = getattr(docstore, "_namespace", "docstore")

            for doc_id, ref_doc_info in ref_docs.items():
                
                if ref_doc_info.metadata and "file_name" in ref_doc_info.metadata:
                    file_name = ref_doc_info.metadata["file_name"]
                    full_path = LOCAL_DOCS_PATH / file_name
                    
                    if not full_path.exists():
                        continue 

                doc_metadata = None
                if ref_doc_info is not None and ref_doc_info.metadata is not None:
                    doc_metadata = IngestedDoc.curate_metadata(ref_doc_info.metadata)
                    
                    if "role" not in ref_doc_info.metadata and kv_store:
                        try:
                            key1 = f"{namespace}/ref_doc_info/{doc_id}"
                            raw_data = kv_store.get(key1)
                            
                            if raw_data is None and namespace == "docstore":
                                key2 = f"ref_doc_info/{doc_id}"
                                raw_data = kv_store.get(key2)

                            if raw_data:
                                if isinstance(raw_data, str):
                                    import json
                                    raw_dict = json.loads(raw_data)
                                else:
                                    raw_dict = raw_data
                                
                                hidden_metadata = raw_dict.get("metadata", {})
                                if "role" in hidden_metadata:
                                    ref_doc_info.metadata.update(hidden_metadata)
                        except Exception as e:
                            pass

                    custom_keys = ["role", "is_infinite", "valid_from", "valid_to"]
                    for key in custom_keys:
                        if key in ref_doc_info.metadata:
                            doc_metadata[key] = ref_doc_info.metadata[key]

                if doc_metadata:
                    ingested_docs.append(
                        IngestedDoc(
                            object="ingest.document",
                            doc_id=doc_id,
                            doc_metadata=doc_metadata,
                        )
                    )
        except Exception as e:
            print(f"Error listando documentos: {e}")
            return []
            
        return ingested_docs

    def delete(self, doc_id: str) -> None:
        logger.info(f"🗑️ START DELETING document={doc_id} from all stores...")
        
        try:
            self.storage_context.vector_store.delete(doc_id)
            logger.info(f"✅ [1/5] Standard VectorStore delete called for: {doc_id}")
        except Exception as e:
            logger.error(f"❌ Error standard vector delete: {e}")

        try:
            if hasattr(self.storage_context.vector_store, "_client"):
                client = self.storage_context.vector_store._client
                collection_name = self.storage_context.vector_store.collection_name
                
                logger.info(f"🧨 Executing DIRECT Qdrant cleanup for doc_id: {doc_id}")

                client.delete(
                    collection_name=collection_name,
                    points_selector=rest.FilterSelector(
                        filter=rest.Filter(
                            must=[
                                rest.FieldCondition(
                                    key="doc_id",
                                    match=rest.MatchValue(value=doc_id)
                                )
                            ]
                        )
                    )
                )

                client.delete(
                    collection_name=collection_name,
                    points_selector=rest.FilterSelector(
                        filter=rest.Filter(
                            must=[
                                rest.FieldCondition(
                                    key="ref_doc_id",
                                    match=rest.MatchValue(value=doc_id)
                                )
                            ]
                        )
                    )
                )
                logger.info(f"✅ [2/5] Direct Qdrant points deleted.")
        except Exception as e:
            logger.warning(f"⚠️ Could not execute direct Qdrant delete (maybe wrong client type): {e}")

        try:
            self.storage_context.index_store.delete_index_struct(doc_id)
            logger.info(f"✅ [3/5] Deleted IndexStruct: {doc_id}")
        except Exception as e:
            pass

        docstore = self.storage_context.docstore
        try:
            docstore.delete_ref_doc(doc_id, raise_error=False)
            logger.info(f"✅ [4/5] Deleted RefDoc: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting RefDoc: {e}")

        try:
            if hasattr(docstore, "_kvstore"):
                kv = docstore._kvstore
                namespace = getattr(docstore, "_namespace", "docstore")
                keys_to_nuke = [
                    f"{namespace}/ref_doc_info/{doc_id}",
                    f"ref_doc_info/{doc_id}",
                    f"{namespace}/doc_id_to_struct_id/{doc_id}",
                    f"doc_id_to_struct_id/{doc_id}",
                    f"{namespace}/metadata/{doc_id}",
                ]
                for key in keys_to_nuke:
                    try:
                        kv.delete(key)
                    except Exception:
                        pass
                logger.info(f"✅ [5/5] Deep Clean KVStore finished.")
        except Exception as e:
            logger.error(f"Error manual cleanup Postgres KV: {e}")

    def update_file_roles(self, file_name: str, new_roles: list[str]) -> None:
        self.update_doc_metadata(file_name, new_roles, None, None, True)

    def update_doc_metadata(
        self, 
        doc_id: str, 
        new_roles: list[str],
        valid_from: str | None,
        valid_to: str | None,
        is_infinite: bool,
        db_id: int | None = None,
        access_url: str | None = None
    ) -> None:
        logger.info(f"Updating metadata for doc_id={doc_id}: Roles={new_roles}, DB_ID={db_id}")

        docstore = self.storage_context.docstore
        
        try:
            ref_doc_info = docstore.get_ref_doc_info(doc_id)
        except Exception:
            ref_doc_info = None

        if not ref_doc_info:
            logger.warning(f"Document with doc_id={doc_id} not found in DocStore")
            return

        new_metadata_updates = {
            "role": new_roles,
            "is_infinite": is_infinite,
            "valid_from": valid_from if not is_infinite else None,
            "valid_to": valid_to if not is_infinite else None,
        }

        if db_id is not None:
            new_metadata_updates["db_id"] = db_id
        
        if access_url is not None:
            new_metadata_updates["access_url"] = access_url
        
        if ref_doc_info.metadata is None:
            ref_doc_info.metadata = {}
        ref_doc_info.metadata.update(new_metadata_updates)

        if ref_doc_info.node_ids:
            try:
                nodes = docstore.get_nodes(ref_doc_info.node_ids)
                nodes_to_update = []
                
                for node in nodes:
                    node.metadata.update(new_metadata_updates)
                    
                    if node.embedding is None:
                        try:
                            # Aquí también forzamos el re-embedding
                            node.embedding = self.embedding_model.get_text_embedding(
                                node.get_content(metadata_mode="embed")
                            )
                        except Exception as emb_err:
                            logger.warning(f"⚠️ Error re-calculating embedding for node {node.node_id}: {emb_err}")
                            continue 
                    
                    nodes_to_update.append(node)
                
                if nodes_to_update:
                    docstore.add_documents(nodes_to_update, allow_update=True)
                    
                    try:
                        self.storage_context.vector_store.delete(doc_id)
                    except Exception as del_err:
                        print(f"⚠️ Aviso: No se pudo borrar previo (quizás es nuevo): {del_err}")

                    self.storage_context.vector_store.add(nodes_to_update)
                    
                    logger.info(f"✅ [SUCCESS] Metadata forced-update to {len(nodes_to_update)} nodes in Qdrant.")
            except Exception as e:
                logger.error(f"❌ Failed to propagate metadata to nodes: {e}")

        saved = False
        try:
            if hasattr(docstore, "_kvstore"):
                json_data = ref_doc_info.to_dict()
                namespace = getattr(docstore, "_namespace", None) or "docstore"
                key1 = f"{namespace}/ref_doc_info/{doc_id}"
                docstore._kvstore.put(key1, json_data)
                saved = True
        except Exception as e:
            logger.error(f"KVStore injection failed: {e}")

        if not saved:
            try:
                if hasattr(docstore, "set_ref_doc_info"):
                    docstore.set_ref_doc_info(doc_id, ref_doc_info)
            except Exception as e:
                logger.warning(f"Failed set_ref_doc_info: {e}")

        logger.info("Metadata update logic completed.")