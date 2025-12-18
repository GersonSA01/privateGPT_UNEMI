import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator

from injector import inject, singleton
import json
import re  # Vital para Regex y limpieza de texto
import time # Vital para medir latencia

from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import SentenceTransformerRerank # Vital para precisión
from llama_index.core.storage import StorageContext
from llama_index.core.response_synthesizers import get_response_synthesizer

from llama_index.core.types import TokenGen
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterCondition
from pydantic import BaseModel
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

# PrivateGPT imports
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chunks.chunks_service import Chunk
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. PROMPTS OPTIMIZADOS
# ==============================================================================


ROUTER_SYSTEM_PROMPT = """
ROLE: Intent Classifier & Search Optimizer.
TASK: Analyze the USER QUERY. Output ONLY valid JSON.

### CONTEXT:
- Active Process: {active_process_str}
- History: {history_str}
- Tools: {data_tools}

### CLASSIFICATION LOGIC (HIERARCHY):
1. GREETING: User says "Hola", "Buenos días" ONLY.
2. CRISIS_SUPPORT: Emotional distress, violence, abuse -> CLASSIFY AS "HUMAN_HANDOFF".
3. OFF_TOPIC (STRICT): 
   - Queries about recipes, food, sports, movies, jokes, politics unrelated to the University.
4. FUNCTION (STRICT MATCHING): 
   - Select ONLY if the user's intent MATCHES the purpose of a tool listed in [TOOLS_LIST] EXACTLY.
   - CRITICAL: Do NOT force a match based on partial keywords.
   - Example: If user asks for "Cambio de Sede" and the tool is "Cambio de Carrera", DO NOT select "Cambio de Carrera". Classify as "RAG".
5. RAG: 
   - User asks for info, dates, requirements, regulations.
   - User asks for a process NOT listed in Tools (e.g., "Homologación", "Cambio de Sede" if not in tools).
6. HUMAN_HANDOFF: User asks for a person/agent.
7. AMBIGUOUS: 
   - The query is ACADEMIC but too short/vague to search. 
   - DO NOT use this for non-academic topics.

### SEARCH QUERY OPTIMIZATION RULES (CRITICAL FOR RAG):
Your goal is to generate a `search_query` optimized for Vector Search (Keywords).
1. **EXTRACT CONCEPTS**: Identify academic terms (e.g., "Matrícula", "Nivelación", "Admisión", "Retiro", "Sanción").
2. **CONTEXTUALIZE**: 
   - If user says "PRE", "Curso de nivelación" or "Admisión" -> You MUST add keyword "Nivelación Admisión".
   - If user says "No pude presentarme", "perdí cupo" or "no continué" -> Add keyword "Sanción Segunda Matrícula".
3. **REMOVE NOISE (STRICT)**: 
   - REMOVE references to attachments.
   - REMOVE personal excuses.
   - REMOVE greetings.
4. **FORMAT**: Return a string of keywords, not a full sentence.

### OUTPUT FORMAT (Strict JSON):
{
  "classification": "FUNCTION" | "HUMAN_HANDOFF" | "AMBIGUOUS" | "RAG" | "GREETING" | "OFF_TOPIC",
  "search_query": "<OPTIMIZED KEYWORDS if RAG, otherwise null>",
  "function_name": "<Tool Name OR null>",
  "clarification": "<String in spanish asking for more info ONLY IF AMBIGUOUS, otherwise null>",
  "handoff_message": "<String if HANDOFF OR null>"
}
"""

RAG_EXPERT_PROMPT = """
ROLE: UNEMI Virtual Assistant - You are a friendly agent helping students and teachers.
TASK: Answer the user using ONLY the information from the provided documents.
OUTPUT: Return ONLY valid JSON. No markdown. No explanations.
LANGUAGE: You MUST respond ALWAYS in SPANISH. Use a conversational, friendly, and approachable tone, as if you were a colleague helping another.

### CORE RULES:
1) **CITATIONS (CRITICAL):**
   - You MUST mention the **Document Name** (if present in context) and the **Article/Section** if it appears.
   - Conversational format in Spanish:
     - "Según el [Document Name], en el Artículo X..." or "Según el [Document Name], en la Sección Y..."
   - If there's no article/section number, mention what you have available in a natural way.

2) **NO MADE-UP PROCEDURES:**
   - Do NOT state as FACT that the user can request:
     - deadline extension / reopening / exception / appeal
     unless the document explicitly mentions that mechanism.
   - If the user asks about these and it's NOT in the document:
     - Say conversationally in Spanish: "En la normativa que revisé no encontré información sobre un mecanismo formal para extender el plazo o reabrir el proceso..."
     - Then continue with what IS in the document.

3) **GUIDANCE ALLOWED (BUT MUST BE CLEAR):**
   - You MAY recommend operational next steps (e.g., "contactar al Vicerrectorado", "Dirección de Carrera") ONLY if:
     A) That unit is explicitly mentioned in the document as part of the process, OR
     B) You label it as **general guidance** (not a rule) when not mandated by the document.
   - If using B, be careful (avoid "debes", "garantizado", "te reabrirán").

4) **CONTROLLED INFERENCE:**
   - If the exact scenario is not mentioned but the document has a general rule that reasonably applies
     (deadlines, requirements, force majeure, validation steps), you may adapt it.
   - When inferring, use cautious phrasing in Spanish:
     - "Según lo que establece el reglamento..." / "En general, el reglamento indica..." / "Esto sugiere que..."
   - Never invent timeframes, forms, departments, or approvals not present in the context.

5) **BRIDGE WITH THE USER:**
   - Acknowledge the user's situation in 1 sentence (in Spanish, friendly).
   - Then cite the rule and explain what it implies (in Spanish, conversational).
   - End with next steps that are either:
     - based on the document, OR clearly labeled as "general guidance".

6) **WHEN answer_found IS FALSE:**
   - If you cannot find relevant information in the provided context:
     - Set "answer_found": false
     - Your "response" MUST include (in Spanish, conversational): "¿Deseas que te contacte con mis compañeros humanos?"
     - Conversational example: "Lo siento, en los documentos que revisé no encontré información específica sobre tu consulta. ¿Deseas que te contacte con mis compañeros humanos para que te puedan ayudar mejor?"

### TONE AND STYLE:
- Speak as a friendly and empathetic virtual assistant
- Use "tú" (not "usted" unless context requires it)
- Be clear but conversational
- Avoid unnecessary technical terms
- Show understanding for the student/teacher's situation
- Act like a helpful colleague, not a formal system

### OUTPUT JSON (STRICT):
{
  "response": "<Response in Spanish, conversational and friendly, with citations. If answer_found is false, MUST include the question about contacting humans>",
  "action": "ANSWER",
  "function_name": null,
  "answer_found": true/false,
  "source_ids": [<int>]
}

IMPORTANT: OUTPUT ONLY RAW JSON. START WITH "{". ALL RESPONSES TO THE USER MUST BE IN SPANISH AND CONVERSATIONAL.
"""

# ==============================================================================
# 2. DATA MODELS & UTILS
# ==============================================================================

class Completion(BaseModel):
    response: str
    sources: list[Chunk] | None = None

class CompletionGen(BaseModel):
    response: TokenGen
    sources: list[Chunk] | None = None

class IdTaggingPostprocessor(BaseNodePostprocessor):
    """Agrega el ID del documento al inicio del texto para que el LLM pueda citarlo."""
    log_preview_chars: int = 60

    def _postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle=None) -> list[NodeWithScore]:
        # Logueamos cuántos nodos llegan a esta etapa
        logger.info(f"🧩 IdTaggingPostprocessor START | nodes_in={len(nodes)}")
        
        for i, n in enumerate(nodes):
            raw_text = n.node.text or ""
            md = getattr(n.node, "metadata", {}) or {}
            db_id = md.get("db_id", "MISSING_DB_ID")
            
            # Extraer nombre del documento si está disponible
            title = md.get("document_name") or md.get("file_name") or "Documento"
            
            # Inyectamos el ID y el nombre del documento en el texto
            n.node.text = (
                f"--- DOC: {title} | ID: {db_id} ---\n"
                f"{raw_text}\n"
                f"--- FIN ---"
            )
        return nodes

@dataclass
class PGChatEngineInput:
    system_message: ChatMessage | None = None
    last_message: ChatMessage | None = None
    chat_history: list[ChatMessage] | None = None

    @classmethod
    def from_messages(cls, messages: list[ChatMessage]) -> "PGChatEngineInput":
        working_messages = list(messages)
        system_message = None
        if working_messages and working_messages[0].role == MessageRole.SYSTEM:
            system_message = working_messages.pop(0)
        last_message = None
        if working_messages and working_messages[-1].role == MessageRole.USER:
            last_message = working_messages.pop(-1)
        return cls(system_message=system_message, last_message=last_message, chat_history=working_messages if working_messages else [])

class KeywordRelevanceFilterPostprocessor(BaseNodePostprocessor):
    """Filtro léxico rápido para descartar basura obvia antes del Reranker costoso."""
    min_hit_ratio: float = 0.06
    min_hits: int = 1
    max_nodes: int = 20

    def _keywords(self, query: str) -> list[str]:
        q = (query or "").lower()
        # Palabras clave académicas para dar boost
        boost_phrases = ["matricul", "matrícul", "retiro", "baja", "anul", "carrera", "nivelación", "arancel", "rubro", "deuda", "saldo", "pago"]
        
        # Regex para sacar palabras de más de 4 letras
        words = re.findall(r"[a-záéíóúñ]{4,}", q)
        
        for bp in boost_phrases:
            if bp in q:
                words.append(bp)
        
        seen = set()
        out = []
        for w in words:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out

    def _hit_score(self, keywords: list[str], text: str) -> tuple[int, float]:
        t = (text or "").lower()
        hits = sum(1 for k in keywords if k in t)
        ratio = hits / max(len(keywords), 1)
        return hits, ratio

    def _postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle=None) -> list[NodeWithScore]:
        if not nodes: return nodes
        query_str = ""
        if query_bundle is not None:
            query_str = getattr(query_bundle, "query_str", "") or ""
            
        keywords = self._keywords(query_str)
        if not keywords: return nodes[: self.max_nodes]
        
        kept: list[NodeWithScore] = []
        for n in nodes:
            content = n.node.get_content() or ""
            hits, ratio = self._hit_score(keywords, content)
            if hits >= self.min_hits and ratio >= self.min_hit_ratio:
                kept.append(n)
                
        # Si el filtro es muy estricto y borra todo, devolvemos los originales
        if not kept: return nodes[: self.max_nodes]
        return kept[: self.max_nodes]

# ==============================================================================
# 3. CHAT SERVICE PRINCIPAL
# ==============================================================================

@singleton
class ChatService:
    settings: Settings

    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.embedding_component = embedding_component
        self.vector_store_component = vector_store_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        self.index = VectorStoreIndex.from_vector_store(
            vector_store_component.vector_store,
            storage_context=self.storage_context,
            llm=llm_component.llm,
            embed_model=embedding_component.embedding_model,
            show_progress=True,
        )

        # =================================================================
        # 🔥 PERSISTENCIA EN MEMORIA: Cargamos el Reranker al arrancar
        # =================================================================
        # Como ChatService es @singleton, esto se ejecuta UNA SOLA VEZ
        # cuando arranca PrivateGPT. El modelo queda en RAM para siempre.
        logger.info("⏳ [INIT] Cargando modelo Reranker (BGE-Base) en memoria RAM...")
        t0 = time.perf_counter()
        
        # Esto descarga el modelo (si no existe) y lo deja fijo en la RAM
        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base", 
            top_n=3  # Mantenemos 3 para balance entre precisión y velocidad
        )
        
        logger.info(f"✅ [INIT] Reranker cargado y listo en {time.perf_counter()-t0:.2f}s")

    # --- DETECTOR DE INTENCIÓN (ROUTER) ---
    def _detect_intent(self, query: str, tools_str: str, chat_history: list[ChatMessage], current_process: str | None) -> dict:
        import unicodedata
        
        raw_query = (query or "").strip()
        # Normalización básica para saludos
        def _norm(s: str) -> str:
            return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
        
        nq = _norm(raw_query)
        if nq in {"hola", "buenas", "buenos dias", "buenas tardes", "hey", "saludos"}:
            return {"type": "GREETING", "payload": None}

        if not tools_str or len(tools_str) < 5:
            return {"type": "RAG", "payload": query}

        active_process_str = f'"{current_process}"' if current_process else "NONE"
        history_str = "None"
        if chat_history:
            # Tomar los últimos 6 mensajes (aproximadamente 3 turnos)
            tail = chat_history[-6:]
            parts = []
            for m in tail:
                if m.role == MessageRole.USER:
                    parts.append(f"User: {m.content[:120]}")
                elif m.role == MessageRole.ASSISTANT:
                    parts.append(f"Assistant: {m.content[:120]}")
            if parts:
                history_str = " | ".join(parts)

        formatted_system_prompt = ROUTER_SYSTEM_PROMPT.replace("{data_tools}", tools_str).replace("{history_str}", history_str).replace("{active_process_str}", active_process_str)

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=formatted_system_prompt),
            ChatMessage(role=MessageRole.USER, content=f'QUERY: "{query}"'),
        ]

        try:
            logger.info("🤖 Router (Merged): LLM call start")
            response = self.llm_component.llm.chat(messages)
            content = response.message.content.strip()
            
            # Limpieza para Llama 3.1
            content = content.replace("```json", "").replace("```", "")
            match = re.search(r"\{.*\}", content, re.DOTALL)
            
            if match:
                data = json.loads(match.group(0))
            else:
                logger.warning(f"⚠️ Router JSON no encontrado. Raw: {content}")
                # Fallback seguro: usar la query original si falla el JSON
                return {"type": "RAG", "payload": query}

            cls_type = data.get("classification", "RAG")
            
            # --- CAMBIO IMPORTANTE AQUÍ ---
            # Si es RAG, devolvemos la query optimizada ("search_query") en lugar de null
            if cls_type == "RAG":
                optimized_query = data.get("search_query") or query  # Fallback a original si viene null
                return {"type": "RAG", "payload": optimized_query}
            
            # El resto sigue igual
            if cls_type == "FUNCTION": return {"type": "FUNCTION", "payload": data.get("function_name")}
            if cls_type == "AMBIGUOUS": return {"type": "AMBIGUOUS", "payload": data.get("clarification", "¿Podrías ser más específico?")}
            if cls_type == "HUMAN_HANDOFF": return {"type": "HUMAN_HANDOFF", "payload": data.get("handoff_message", "¿Deseas contactar a un agente?")}
            if cls_type == "GREETING": return {"type": "GREETING", "payload": None}
            if cls_type == "OFF_TOPIC": return {"type": "OFF_TOPIC", "payload": None}
            
            return {"type": "RAG", "payload": query}

        except Exception as e:
            logger.error(f"❌ Router Error: {e}")
            return {"type": "RAG", "payload": query}

    def _is_truthy(self, v) -> bool:
        if isinstance(v, bool): return v
        if v is None: return False
        return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

    # --- MOTOR DE CHAT (PIPELINE RAG) ---
    def _chat_engine(self, system_prompt: str | None = None, use_context: bool = False, context_filter: ContextFilter | None = None) -> BaseChatEngine:
        if use_context:
            final_system_prompt = system_prompt if system_prompt else "Answer using context."
            
            # Construcción de filtros de metadatos (por archivo/DB ID)
            filters = None
            if context_filter and context_filter.docs_ids:
                filters_list = []
                for raw_id in context_filter.docs_ids:
                    try:
                        db_id = int(raw_id)
                        filters_list.append(MetadataFilter(key="db_id", value=db_id))
                    except (ValueError, TypeError): continue
                if filters_list:
                    filters = MetadataFilters(filters=filters_list, condition=FilterCondition.OR)

            # --- ESTRATEGIA DE RECUPERACIÓN OPTIMIZADA ---
            # 1. Recuperación amplia (15 docs) para no perder nada relevante
            vector_index_retriever = self.index.as_retriever(
                similarity_top_k=15, 
                filters=filters,
            )

            # 2. Reranker preciso (BGE-Base) que selecciona solo los 3 mejores
            # ✅ USAMOS EL MODELO QUE YA ESTÁ EN MEMORIA (cargado en __init__)
            # Esto elimina la latencia de ~1.5 minutos de carga en cada consulta

            node_postprocessors: list[BaseNodePostprocessor] = [
                MetadataReplacementPostProcessor(target_metadata_key="window"),
                KeywordRelevanceFilterPostprocessor(min_hit_ratio=0.06, min_hits=1, max_nodes=10),
                self.reranker,  # <-- Reutilizamos el modelo en memoria (latencia 0s)
                IdTaggingPostprocessor(), # <-- Etiquetado para citas
            ]

            # 3. Sintetizador de respuesta COMPACTO y de UNA SOLA LLAMADA
            # IMPORTANTE: Pasamos 'llm' aquí para evitar el error de OpenAI
            response_synthesizer = get_response_synthesizer(
                llm=self.llm_component.llm,
                response_mode="compact", # Evita 'Refine' (doble llamada)
                streaming=True,
            )

            return ContextChatEngine.from_defaults(
                system_prompt=final_system_prompt,
                retriever=vector_index_retriever,
                llm=self.llm_component.llm,
                node_postprocessors=node_postprocessors,
                response_synthesizer=response_synthesizer,
            )

        # Si no usa contexto (ej. preguntas generales no RAG)
        return SimpleChatEngine.from_defaults(
            system_prompt=system_prompt,
            llm=self.llm_component.llm,
        )

    def _mock_stream_generator(self, text: str) -> Generator[str, None, None]:
        yield text

    def _wrap_timed_generator(self, gen: Generator[str, None, None], label: str) -> Generator[str, None, None]:
        """Envuelve el generador para medir tiempos de respuesta (TTFT y Total)."""
        t0 = time.perf_counter()
        first = True
        n_tokens = 0
        try:
            for tok in gen:
                n_tokens += 1
                if first:
                    first = False
                    logger.info(f"⏱️ {label} first_token after {time.perf_counter()-t0:.3f}s")
                yield tok
        finally:
            logger.info(f"⏱️ {label} done | tokens={n_tokens} | total={time.perf_counter()-t0:.3f}s")

    # --- MÉTODO PÚBLICO PRINCIPAL ---
    def stream_chat(self, messages: list[ChatMessage], use_context: bool = False, context_filter: ContextFilter | None = None) -> CompletionGen:
        chat_engine_input = PGChatEngineInput.from_messages(messages)
        user_query = chat_engine_input.last_message.content if chat_engine_input.last_message else ""
        
        is_reformulation_mode = False
        is_rag_expert_mode = False

        # Detectar Banderas (Flags) del Sistema
        if chat_engine_input.system_message:
            content = chat_engine_input.system_message.content
            if content == "RAG_EXPERT_MODE":
                logger.info("⚖️ Flag detected: RAG_EXPERT_MODE")
                is_rag_expert_mode = True
                safe_prompt = RAG_EXPERT_PROMPT.replace("{", "{{").replace("}", "}}")
                
                chat_engine_input.system_message.content = safe_prompt
                use_context = True

        # Router Logic (Si no hay banderas)
        if not is_reformulation_mode and not is_rag_expert_mode:
            data_tools = ""
            if chat_engine_input.system_message:
                raw_system = chat_engine_input.system_message.content
                # Extracción de herramientas pasadas por Django
                current_process_val: str | None = None
                data_tools = raw_system
                marker_start = data_tools.find("[ACTIVE_PROCESS]")
                marker_end = data_tools.find("[/ACTIVE_PROCESS]", marker_start + 1)
                if marker_start != -1 and marker_end != -1:
                    inside = data_tools[marker_start + len("[ACTIVE_PROCESS]") : marker_end]
                    current_process_val = inside.strip().strip('"')
                    data_tools = data_tools[:marker_start] + data_tools[marker_end + len("[/ACTIVE_PROCESS]") :]

                if "[TOOLS_LIST]" in data_tools:
                    router_history = chat_engine_input.chat_history or []
                    intent_result = self._detect_intent(user_query, data_tools, router_history, current_process=current_process_val)
                    intent_type = intent_result["type"]
                    payload = intent_result["payload"]
                    
                    if intent_type == "GREETING":
                        json_resp = '{"response": "¡Hola! 👋 Soy el asistente virtual de la UNEMI.", "action": "ANSWER", "function_name": null}' 
                        return CompletionGen(response=self._mock_stream_generator(json_resp), sources=[])
                    if intent_type == "OFF_TOPIC":
                        json_resp = '{"response": "Solo temas académicos.", "action": "ANSWER", "function_name": "OFF_TOPIC"}'
                        return CompletionGen(response=self._mock_stream_generator(json_resp), sources=[])
                    if intent_type == "HUMAN_HANDOFF":
                        mock = {"response": payload, "action": "FUNCTION", "function_name": "HUMAN_HANDOFF"}
                        return CompletionGen(response=self._mock_stream_generator(json.dumps(mock, ensure_ascii=False)), sources=[])
                    if intent_type == "AMBIGUOUS":
                        mock = {"response": payload, "action": "ANSWER", "function_name": None}
                        return CompletionGen(response=self._mock_stream_generator(json.dumps(mock, ensure_ascii=False)), sources=[])
                    if intent_type == "FUNCTION": 
                        json_resp = f'{{"response": "Procesando...", "action": "FUNCTION", "function_name": "{payload}"}}'
                        return CompletionGen(response=self._mock_stream_generator(json_resp), sources=[])

                    # CASO RAG:
                    # El payload ahora es la query optimizada (ej: "Normativa matricula extemporanea")
                    # La enviamos en el campo "reformulated_query" para que el frontend la use directamente
                    json_resp = json.dumps({
                        "response": "RAG_MODE", 
                        "action": "ANSWER", 
                        "function_name": None,
                        "reformulated_query": payload  # <--- ENVIAMOS LA QUERY LISTA
                    }, ensure_ascii=False)
                    
                    return CompletionGen(response=self._mock_stream_generator(json_resp), sources=[])

        if use_context:
            logger.info(f"🔎 RAG Query: '{user_query}'")

        chat_history = chat_engine_input.chat_history if chat_engine_input.chat_history else None
        target_system_prompt = None
        if chat_engine_input.system_message:
            target_system_prompt = chat_engine_input.system_message.content

        # Inicializar motor de chat (Aquí ocurre la magia de Rerank + Compact)
        chat_engine = self._chat_engine(
            system_prompt=target_system_prompt, 
            use_context=use_context,
            context_filter=context_filter 
        )
        
        t_sc0 = time.perf_counter()
        logger.info(f"⏱️ chat_engine.stream_chat START | use_context={use_context} | q_len={len(user_query)}")

        # Llamada bloqueante al LLM (Aquí es donde se demora)
        streaming_response = chat_engine.stream_chat(message=user_query, chat_history=chat_history)

        t_sc1 = time.perf_counter()
        logger.info(f"⏱️ chat_engine.stream_chat RETURNED | elapsed={t_sc1-t_sc0:.3f}s")

        t_src0 = time.perf_counter()
        logger.info("⏱️ sources build START")
        try:
            sources = [Chunk.from_node(node) for node in (streaming_response.source_nodes or [])]
        except Exception as e:
            logger.exception(f"❌ sources build failed: {e}")
            sources = []
        t_src1 = time.perf_counter()
        logger.info(f"⏱️ sources build END | sources={len(sources)} | elapsed={t_src1-t_src0:.3f}s")

        wrapped_gen = self._wrap_timed_generator(streaming_response.response_gen, label=("RAG" if use_context else "CHAT"))

        return CompletionGen(response=wrapped_gen, sources=sources)

    @staticmethod
    def _extract_last_json_object(text: str) -> str:
        """Extrae el último objeto JSON válido para limpiar respuestas sucias."""
        if not text: return text
        cleaned = text.strip().replace("```json", "").replace("```", "").strip()
        try:
            json.loads(cleaned)
            return cleaned
        except Exception: pass
        last_open = cleaned.rfind("{")
        last_close = cleaned.rfind("}")
        if last_open != -1 and last_close != -1 and last_close > last_open:
            candidate = cleaned[last_open:last_close + 1].strip()
            try:
                json.loads(candidate)
                return candidate
            except Exception: pass
        return cleaned

    def chat(self, messages: list[ChatMessage], use_context: bool = False, context_filter: ContextFilter | None = None) -> Completion:
        gen = self.stream_chat(messages, use_context, context_filter)
        full_resp_parts: list[str] = []
        for token in gen.response:
            full_resp_parts.append(token)
        full_resp = "".join(full_resp_parts)
        full_resp = self._extract_last_json_object(full_resp)
        return Completion(response=full_resp, sources=gen.sources)