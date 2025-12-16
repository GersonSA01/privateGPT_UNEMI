import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator

from injector import inject, singleton
from llama_index.core.chat_engine import ContextChatEngine, SimpleChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.storage import StorageContext
from llama_index.core.types import TokenGen
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterCondition
from pydantic import BaseModel
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
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
# 1. PROMPTS (LA FUENTE DE LA VERDAD)
# ==============================================================================

ROUTER_SYSTEM_PROMPT = """
ROLE: UNEMI Query Classifier & UX Specialist.
GOAL: Categorize user input. Be EXTREMELY CONSERVATIVE with Function Calling.

### AVAILABLE TOOLS:
{data_tools}

### CLASSIFICATION RULES:

1) FUNCTION (Specific Administrative Actions)
   - ONLY if user explicitly asks to PERFORM an action like "Change career", "Withdraw subject".
   - IF user asks "How do I withdraw?", it is RAG (Information), NOT Function.

2) HUMAN_HANDOFF
   - Keywords: "asesor", "persona", "humano", "contactar agente".

3) GREETING
   - "Hola", "Buenos días", "Hey".

4) OFF_TOPIC (Critical)
   - Any query unrelated to University, Academics, Regulations, or Administrative processes.
   - Examples: "Quiero comer", "Cuéntame un chiste", "Precio del Bitcoin", "Receta de cocina".
   - ACTION: Classify as OFF_TOPIC immediately.

5) RAG (The Fallback)
   - Use this for EVERYTHING related to the University that is not a direct Function action.
   - Complaints, questions about grades, dates, regulations, locations, admission.

### OUTPUT SCHEMA (Strict JSON):
{
  "classification": "FUNCTION" | "HUMAN_HANDOFF" | "AMBIGUOUS" | "RAG" | "GREETING" | "OFF_TOPIC",
  "function_name": "<Tool Name OR null>",
  "clarification": "<String OR null>",
  "handoff_message": "<String OR null>"
}

IMPORTANT:
- DO NOT use Markdown formatting (no ```json).
- Output ONLY the JSON object.
- If you include any extra text (headings, reasoning, markdown), you FAILED.

"""


RAG_EXPERT_PROMPT = """
SYSTEM: UNEMI AGENT BOT (Regulations Expert).
ROLE: You analyze UNEMI documents and regulations to answer the user.
OUTPUT FORMAT: RAW JSON (No Markdown, No Backticks).

### CONTEXT:
The user provides chunks separated by `--- COMIENZO DEL DOCUMENTO [ID: <number>] ---`.

### RULES:
1) IGNORE information that is not explicitly in the provided context.
0) NEVER invent steps, requirements, fees, deadlines, or procedures that are not explicitly stated in the context.
   - If the context does not mention a topic asked by the user, you MUST say you don't have that information.
   - Use wording like: "En los fragmentos proporcionados no encuentro..."
   - Only assert something if you can point to it in the context.


2) If the context talks about "Safety" and the user asks about "Pizza", answer that you don't have that info.
3) "source_ids" must match
- Set "answer_found" to true ONLY if the context contains DIRECT information that answers the user's question.
  If you can only provide partial info (e.g., you found something about 'cambio de carrera' but nothing about 'baja de matrícula'), set answer_found=false and clearly say what you found vs what you did not.
 the ID in the header of the chunk used.

### JSON STRUCTURE:
{{
  "response": "<Your answer in Spanish>",
  "action": "ANSWER",
  "function_name": null,
  "answer_found": <true/false>,
  "source_ids": [<int>, <int>]
}}

### EXAMPLES:

Case 1: Info Found
User: "¿Cuántas matrículas existen?"
Context: "...existen tres matrículas: ordinaria, extraordinaria..." [ID: 10]
Output:
{{
  "response": "Según el reglamento, existen tres tipos de matrículas: ordinaria, extraordinaria y especial.",
  "action": "ANSWER",
  "function_name": null,
  "answer_found": true,
  "source_ids": [10]
}}

Case 2: Info NOT Found
User: "Quiero pizza"
Context: "...reglamento de higiene..." [ID: 20]
Output:
{{
  "response": "Lo siento, los documentos proporcionados no contienen información sobre venta de alimentos o pizza.",
  "action": "ANSWER",
  "function_name": null,
  "answer_found": false,
  "source_ids": []
}}

IMPORTANT: OUTPUT RAW JSON ONLY. DO NOT START WITH "Here is the JSON". START WITH "{{".
"""



REFORMULATION_SYSTEM_PROMPT = """
ROLE: Expert Search Query Optimizer for University Regulations.
TASK: Convert the user's raw input into a precise REGULATORY SEARCH QUERY.

### RULES:
1) Analyze the problem: blocked process, missed deadline, technical error, requirements, etc.
2) Map to institutional terminology:
   - "No alcancé" / "Se cerró" -> search: "Solicitud extemporánea" OR "Plazos vencidos"
   - "Me olvidé" -> search: "Sanción incumplimiento" OR "Justificación"
   - "Quiero ser ayudante" -> search: "Requisitos ayudantía"
3) Remove noise: greetings, names, emotional pleas, unrelated text.
4) OUTPUT: RETURN ONLY THE SEARCH QUERY IN SPANISH. NO EXPLANATIONS. NO QUOTES.

### EXAMPLES:
Input: "Hola, chuta no alcancé a subir los papeles de la beca y ya no me deja"
Output: Normativa solicitud extemporánea documentos de beca consecuencias

Input: "Soy profesor y no sé cómo evaluar el segundo parcial"
Output: Reglamento evaluación y calificación docente

Input: "USER QUERY: {query}"
Output:
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
    def _postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle=None) -> list[NodeWithScore]:
        print("\n🔎 --- DEBUG: INSPECCIONANDO NODOS RECUPERADOS ---")
        for i, n in enumerate(nodes):
            # Imprimimos TODAS las claves de metadata para ver qué diablos hay dentro
            print(f"📦 Nodo #{i} Metadata Keys: {list(n.node.metadata.keys())}")
            print(f"   📄 Content Preview: {n.node.text[:50]}...")
            
            # Verificamos valor exacto
            val_db_id = n.node.metadata.get("db_id")
            print(f"   🆔 Valor de 'db_id': {val_db_id} (Tipo: {type(val_db_id)})")

            # Si db_id no existe, intentamos usar 'file_name' como fallback
            db_id = val_db_id
            if not db_id:
                db_id = "ERROR_METADATA" 
            
            # Formato más explícito para el LLM
            n.node.text = f"--- COMIENZO DEL DOCUMENTO [ID_BASE_DATOS: {db_id}] ---\n{n.node.text}\n--- FIN FRAGMENTO ---"
        print("--------------------------------------------------\n")
        return nodes

@dataclass
class PGChatEngineInput:
    system_message: ChatMessage | None = None
    last_message: ChatMessage | None = None
    chat_history: list[ChatMessage] | None = None

    # --- MOVER ESTE MÉTODO AQUÍ ADENTRO ---
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

# Asegúrate de tener este import al inicio del archivo junto a los otros
import re 

# ... (resto de imports)

class KeywordRelevanceFilterPostprocessor(BaseNodePostprocessor):
    """
    Lightweight lexical relevance filter to reduce obviously off-topic nodes.
    """
    # Definición de campos estilo Pydantic (LlamaIndex v0.10+)
    min_hit_ratio: float = 0.06
    min_hits: int = 1
    max_nodes: int = 20

    def _keywords(self, query: str) -> list[str]:
        q = (query or "").lower()
        # Commonly relevant stems for UNEMI academic processes
        boost_phrases = [
            "matricul", "matrícul", "retiro", "baja", "anul", "carrera", "nivelación",
            "arancel", "rubro", "deuda", "saldo", "pago",
        ]
        # NOTA: Asegúrate de importar 're' al inicio del archivo
        words = re.findall(r"[a-záéíóúñ]{4,}", q)
        # Add stems from boost phrases if present in query
        for bp in boost_phrases:
            if bp in q:
                words.append(bp)
        # Deduplicate while preserving order
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

    # CORRECCIÓN AQUÍ: Se agregó el guion bajo al inicio (_)
    def _postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle=None) -> list[NodeWithScore]:
        if not nodes:
            return nodes

        query_str = ""
        if query_bundle is not None:
            query_str = getattr(query_bundle, "query_str", "") or ""

        keywords = self._keywords(query_str)
        if not keywords:
            return nodes[: self.max_nodes]

        kept: list[NodeWithScore] = []
        for n in nodes:
            content = ""
            try:
                content = n.node.get_content(metadata_mode="none")
            except Exception:
                try:
                    content = n.node.get_content()
                except Exception:
                    content = ""
            hits, ratio = self._hit_score(keywords, content)
            if hits >= self.min_hits and ratio >= self.min_hit_ratio:
                kept.append(n)

        # Never drop everything; fall back to original top nodes if filter is too strict.
        if not kept:
            return nodes[: self.max_nodes]

        return kept[: self.max_nodes]

# ==============================================================================
# 3. CHAT SERVICE
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


    def _detect_intent(self, query: str, tools_str: str) -> dict:
        """
        Router Simplificado: Delega el 100% de la decisión al LLM.
        Se eliminaron las validaciones por keywords/score para evitar falsos positivos.
        """
        import re
        import unicodedata
        import json

        raw_query = (query or "").strip()

        # 1. Normalización básica (solo para detectar saludos vacíos rápidamente)
        def _norm(s: str) -> str:
            s = (s or "").lower().strip()
            s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
            return s

        def _is_pure_greeting(nq: str) -> bool:
            # Mantenemos esto solo para ahorrar tokens/dinero en "holas" simples
            return nq in {"hola", "buenas", "buenos dias", "buenas tardes", "buenas noches", "hey", "saludos"}

        nq = _norm(raw_query)

        # A. Si es solo un saludo, retornamos rápido (ahorro de recursos)
        if _is_pure_greeting(nq):
            return {"type": "GREETING", "payload": None}
            
        # B. Si no hay herramientas disponibles, es RAG directo
        if not tools_str or len(tools_str) < 5:
            return {"type": "RAG", "payload": None}
        
        formatted_system_prompt = ROUTER_SYSTEM_PROMPT.replace("{data_tools}", tools_str)

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=formatted_system_prompt),
            ChatMessage(role=MessageRole.USER, content=f'USER QUERY: "{query}"'),
        ]

        try:
            # Llamada al modelo
            response = self.llm_component.llm.chat(messages)
            content = response.message.content.strip()
            
            # --- CORRECCIÓN ROBUSTA ---
            # Buscamos dónde empieza el primer '{' y dónde termina el último '}'
            start_index = content.find('{')
            end_index = content.rfind('}')

            if start_index != -1 and end_index != -1:
                clean_content = content[start_index : end_index + 1]
            else:
                logger.warning("Router Output no contiene JSON válido. Forzando RAG.")
                return {"type": "RAG", "payload": None}
            
            # Parseamos el JSON limpio
            data = json.loads(clean_content)

            classification = data.get("classification")
            function_name = data.get("function_name")
            clarification = data.get("clarification")
            handoff_message = data.get("handoff_message")

            # --- PARSEO DE RESPUESTA DEL LLM ---

            if classification == "FUNCTION" and function_name:
                return {"type": "FUNCTION", "payload": function_name}

            if classification == "AMBIGUOUS":
                # Si el LLM dice que es ambiguo (ej: user puso "carrera"), devolvemos su pregunta aclaratoria
                msg = clarification if isinstance(clarification, str) and clarification.strip() else "Por favor, sé más específico con tu solicitud."
                return {"type": "AMBIGUOUS", "payload": msg}

            if classification == "GREETING":
                return {"type": "GREETING", "payload": None}

            if classification == "OFF_TOPIC":
                return {"type": "OFF_TOPIC", "payload": None}


            if classification == "HUMAN_HANDOFF":
                msg = handoff_message if isinstance(handoff_message, str) and handoff_message.strip() else \
                      "Entiendo. ¿Quieres que te conecte con un agente humano? Si es posible, cuéntame brevemente tu caso y adjunta evidencia (opcional)."
                return {"type": "HUMAN_HANDOFF", "payload": msg}
   

            # Default: RAG
            return {"type": "RAG", "payload": None}

        except Exception as e:
            logger.error(f"Router LLM Error: {e}")
            # En caso de error técnico del LLM, fallback seguro a RAG
            return {"type": "RAG", "payload": None}


    def _chat_engine(
        self,
        system_prompt: str | None = None,
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> BaseChatEngine:
        
        if use_context:
            final_system_prompt = system_prompt if system_prompt else "Answer using context."
            
            filters = None
            if context_filter and context_filter.docs_ids:
                filters_list = []
                for raw_id in context_filter.docs_ids:
                    try:
                        # Asegúrate que esta línea esté así.
                        # Convertimos a int porque en Qdrant guardaste db_id como entero.
                        db_id = int(raw_id) 
                        filters_list.append(MetadataFilter(key="db_id", value=db_id))
                    except (ValueError, TypeError):
                        # Si viene basura, lo ignoramos para no romper el chat
                        continue

                if filters_list:
                    filters = MetadataFilters(
                        filters=filters_list,
                        condition=FilterCondition.OR
                    )
                else:
                    filters = None

            vector_index_retriever = self.index.as_retriever(
                similarity_top_k=15, 
                filters=filters  
            )
            
            # === CAMBIO AQUÍ: AÑADIMOS EL ID TAGGER ===
            node_postprocessors: list[BaseNodePostprocessor] = [
                MetadataReplacementPostProcessor(target_metadata_key="window"),
                KeywordRelevanceFilterPostprocessor(min_hit_ratio=0.06, min_hits=1, max_nodes=15),
                IdTaggingPostprocessor(), # <--- NUESTRO NUEVO PROCESADOR
            ]
            
            if self.settings.rag.rerank.enabled:
                rerank_postprocessor = SentenceTransformerRerank(
                    model=self.settings.rag.rerank.model, top_n=8
                )
                node_postprocessors.append(rerank_postprocessor)

            return ContextChatEngine.from_defaults(
                system_prompt=final_system_prompt,
                retriever=vector_index_retriever,
                llm=self.llm_component.llm,
                node_postprocessors=node_postprocessors,
                context_mode="compact" 
            )
        else:
            return SimpleChatEngine.from_defaults(
                system_prompt=system_prompt,
                llm=self.llm_component.llm,
            )

    def _mock_stream_generator(self, text: str) -> Generator[str, None, None]:
        yield text

    # === LÓGICA PRINCIPAL CON BANDERAS ===
    def stream_chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> CompletionGen:
        
        chat_engine_input = PGChatEngineInput.from_messages(messages)
        user_query = chat_engine_input.last_message.content if chat_engine_input.last_message else ""
        
        # ==============================================================================
        # 🧪 ZONA DE PRUEBAS (MOCKS AVANZADOS)
        # ==============================================================================
        import json

        # CASO 1: RAG EXITOSO (Simula que encontró normativa y IDs de documentos)
        # Úsalo para ver si Django convierte los IDs [10, 12] en enlaces PDF.
        if "#TEST_RAG_FOUND" in user_query:
            mock_response = {
                "response": "Según el Artículo 45 del Reglamento de Estudiantes, se permite la tercera matrícula bajo condiciones especiales de calamidad doméstica.",
                "action": "ANSWER",
                "function_name": None,
                "answer_found": True,
                "source_ids": [10, 12]
            }
            return CompletionGen(response=self._mock_stream_generator(json.dumps(mock_response)), sources=[])

        # CASO 2: RAG FALLIDO (Simula que leyó pero no encontró nada)
        if "#TEST_RAG_EMPTY" in user_query:
            mock_response = {
                "response": "He revisado la normativa vigente pero no encontré información específica sobre 'Viajes a Marte'.",
                "action": "ANSWER",
                "function_name": None,
                "answer_found": False,
                "source_ids": []
            }
            return CompletionGen(response=self._mock_stream_generator(json.dumps(mock_response)), sources=[])

        # CASO 3: SIMULAR FUNCIÓN DETECTADA (Para probar el flujo de "Inactivo")
        # Django recibirá esto. Si en tu BD este proceso está 'status=False' o fechas vencidas,
        # Django debería rechazarlo y mandar al RAG automáticamente.
        if "#TEST_FUNC_DETECTADA" in user_query:
            mock_response = {
                "classification": "FUNCTION",
                # Asegúrate de poner aquí un nombre que tengas en BD como INACTIVO o CADUCADO
                "function_name": "Retiro de materia ordinaria", 
                "action": "FUNCTION",
                "response": "Procesando..."
            }
            return CompletionGen(response=self._mock_stream_generator(json.dumps(mock_response)), sources=[])

        # CASO 4: SIMULAR AMBIGÜEDAD (El que ya tenías)
        if "#TEST_AMBIGUO" in user_query:
            mock_response = {
                "classification": "AMBIGUOUS",
                "function_name": None,
                "clarification": "¿Te refieres a Justificación Médica o Justificación Laboral?",
                "action": "ANSWER",
                "response": "¿Te refieres a Justificación Médica o Justificación Laboral?"
            }
            return CompletionGen(response=self._mock_stream_generator(json.dumps(mock_response)), sources=[])
        
        if "#TEST_HANDOFF" in user_query:
            mock_response = {
                "classification": "HUMAN_HANDOFF",
                "function_name": None,
                "handoff_message": "Perfecto. Puedo pasarte con un agente humano. ¿Deseas continuar con el contacto? (Sí/No)",
                "action": "HUMAN_HANDOFF",
                "response": "Perfecto. Puedo pasarte con un agente humano. ¿Deseas continuar con el contacto? (Sí/No)"
            }
            return CompletionGen(response=self._mock_stream_generator(json.dumps(mock_response)), sources=[])

        # 1. GESTIÓN DE BANDERAS (FLAGS)
        is_reformulation_mode = False
        is_rag_expert_mode = False

        if chat_engine_input.system_message:
            content = chat_engine_input.system_message.content
            
            # A. Bandera de Reformulación
            if content == "REFORMULATE_QUERY_MODE":
                logger.info("🔄 Flag detected: REFORMULATE_QUERY_MODE")
                is_reformulation_mode = True
                chat_engine_input.system_message.content = REFORMULATION_SYSTEM_PROMPT
                use_context = False 
            
            # B. Bandera de Experto Legal (Nuevo)
            elif content == "RAG_EXPERT_MODE":
                logger.info("⚖️ Flag detected: RAG_EXPERT_MODE")
                is_rag_expert_mode = True
                # AHORA SÍ EXISTE LA VARIABLE
                chat_engine_input.system_message.content = RAG_EXPERT_PROMPT
                use_context = True 

        # 2. LÓGICA DE ROUTER (Solo si no hay banderas especiales)
        if not is_reformulation_mode and not is_rag_expert_mode:
            data_tools = ""
            if chat_engine_input.system_message:
                data_tools = chat_engine_input.system_message.content
                if "[TOOLS_LIST]" in data_tools:
                    # === CAMBIO AQUÍ ===
                    intent_result = self._detect_intent(user_query, data_tools)
                    intent_type = intent_result["type"]
                    payload = intent_result["payload"]
                    
                    if intent_type == "GREETING":
                        json_resp = '{"response": "¡Hola! 👋 Soy el asistente virtual de la UNEMI.", "action": "ANSWER", "function_name": null}' 
                        return CompletionGen(response=self._mock_stream_generator(json_resp), sources=[])
                    
                    if intent_type == "OFF_TOPIC":
                        json_resp = '{"response": "Solo temas académicos.", "action": "ANSWER", "function_name": "OFF_TOPIC"}'
                        return CompletionGen(response=self._mock_stream_generator(json_resp), sources=[])
                    
                    # === NUEVO: HUMAN HANDOFF ===
                    if intent_type == "HUMAN_HANDOFF":
                        mock = {"response": payload, "action": "FUNCTION", "function_name": "HUMAN_HANDOFF"}
                        json_resp = json.dumps(mock, ensure_ascii=False)
                        return CompletionGen(response=self._mock_stream_generator(json_resp), sources=[])

                    # === MANEJO DE AMBIGÜEDAD (FIX: era text_resp) ===
                    if intent_type == "AMBIGUOUS":
                        mock = {"response": payload, "action": "ANSWER", "function_name": None}
                        json_resp = json.dumps(mock, ensure_ascii=False)
                        return CompletionGen(response=self._mock_stream_generator(json_resp), sources=[])


                    if intent_type == "FUNCTION": 
                        json_resp = f'{{"response": "Procesando...", "action": "FUNCTION", "function_name": "{payload}"}}'
                        return CompletionGen(response=self._mock_stream_generator(json_resp), sources=[])

                    # Si es RAG normal
                    json_resp = '{"response": "RAG_MODE", "action": "ANSWER", "function_name": null}'
                    return CompletionGen(response=self._mock_stream_generator(json_resp), sources=[])

        # 3. EJECUCIÓN
        if use_context:
            logger.info(f"🔎 RAG Query: '{user_query}'")

        chat_history = chat_engine_input.chat_history if chat_engine_input.chat_history else None
        
        target_system_prompt = None
        if chat_engine_input.system_message:
            target_system_prompt = chat_engine_input.system_message.content

        chat_engine = self._chat_engine(
            system_prompt=target_system_prompt, 
            use_context=use_context,
            context_filter=context_filter 
        )
        
        streaming_response = chat_engine.stream_chat(
            message=user_query, 
            chat_history=chat_history,
        )
        sources = [Chunk.from_node(node) for node in streaming_response.source_nodes]

        return CompletionGen(
            response=streaming_response.response_gen, sources=sources
        )

    @staticmethod
    def _extract_last_json_object(text: str) -> str:
        """
        Best-effort extraction of the *last* JSON object from a model output.
        This protects callers when the underlying LlamaIndex response synthesizer
        produces multiple JSON answers (e.g., CompactAndRefine/Refine).
        """
        if not text:
            return text
        cleaned = text.strip().replace("```json", "").replace("```", "").strip()
        # Fast path: already valid JSON
        try:
            json.loads(cleaned)
            return cleaned
        except Exception:
            pass
        # Common case: multiple JSON objects concatenated -> keep the last one
        last_open = cleaned.rfind("{")
        last_close = cleaned.rfind("}")
        if last_open != -1 and last_close != -1 and last_close > last_open:
            candidate = cleaned[last_open:last_close + 1].strip()
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass
        # Try splitting on likely boundaries
        for sep in ("}\n{", "}\r\n{", "}{"):
            idx = cleaned.rfind(sep)
            if idx != -1:
                candidate = "{" + cleaned[idx + len(sep):] if sep != "}{" else cleaned[idx+1:]
                candidate = candidate.strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        continue
        return cleaned

    def chat(self, messages: list[ChatMessage], use_context: bool = False, context_filter: ContextFilter | None = None) -> Completion:
        # NOTE: stream_chat() can yield *multiple* answer chunks when LlamaIndex uses refine-mode synthesis.
        # We concatenate the stream, then keep only the last valid JSON object to avoid 'double answers'.
        gen = self.stream_chat(messages, use_context, context_filter)
        full_resp_parts: list[str] = []
        for token in gen.response:
            full_resp_parts.append(token)
        full_resp = "".join(full_resp_parts)
        full_resp = self._extract_last_json_object(full_resp)
        return Completion(response=full_resp, sources=gen.sources)
