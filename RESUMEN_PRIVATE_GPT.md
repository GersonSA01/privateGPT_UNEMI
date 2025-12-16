# Resumen Técnico Detallado: Private-GPT Main

Este documento ofrece un análisis profundo de la arquitectura interna de `private-gpt-main`. Este proyecto es una API robusta construida sobre **FastAPI** y **LlamaIndex** que permite la ingestión de documentos y la generación de respuestas aumentadas por recuperación (RAG) de manera local y privada.

## 🏗️ Arquitectura de Alto Nivel

El sistema sigue una arquitectura modular basada en **Inyección de Dependencias** (usando la librería `injector`). Esto permite cambiar fácilmente entre diferentes implementaciones de LLM (Ollama, OpenAI, LlamaCPP) y bases de datos vectoriales (Qdrant, Chroma, Postgres).

### Componentes Principales

1.  **Server (FastAPI):** Expone endpoints REST compatibles con la API de OpenAI.
2.  **Core (LlamaIndex):** Orquesta la lógica de RAG, embeddings y chat.
3.  **Components:** Módulos intercambiables para LLM, Embeddings y Vector Store.

---

## 📂 Análisis Archivo por Archivo

A continuación, se destacan los archivos más críticos del sistema.

### 1. Configuración y Arranque

#### 📄 `settings.yaml` (El Cerebro de Configuración)
**Importancia:** ⭐⭐⭐⭐⭐
Define el comportamiento global del sistema.
*   **`ui.default_query_system_prompt`:** Aquí reside el prompt maestro que instruye a la IA sobre cómo comportarse (ej: reglas de "has_information", "needs_contact").
*   **`llm` y `ollama`:** Configura el modelo a usar (ej: `llama3.1`), la ventana de contexto y la temperatura.
*   **`vectorstore`:** Define qué base de datos usar (actualmente `qdrant`).

#### 📄 `private_gpt/launcher.py`
**Importancia:** ⭐⭐⭐⭐
Es la fábrica de la aplicación FastAPI.
*   Configura el contenedor de inyección de dependencias (`Injector`).
*   Registra los routers (`chat_router`, `ingest_router`, etc.).
*   Configura CORS y monta la UI si está habilitada.

### 2. Capa de Servicio (Lógica de Negocio)

#### 📄 `private_gpt/server/chat/chat_router.py`
**Importancia:** ⭐⭐⭐⭐
El punto de entrada para las peticiones de chat (`/v1/chat/completions`).
*   Recibe el JSON del usuario.
*   Delega la lógica al `ChatService`.
*   Maneja el streaming de respuestas (SSE) para que el texto aparezca "escribiéndose".

#### 📄 `private_gpt/server/chat/chat_service.py` (El Orquestador RAG)
**Importancia:** ⭐⭐⭐⭐⭐ (Crítica)
Aquí ocurre la magia del RAG.
*   **Clase `ChatService`:** Inicializa el `VectorStoreIndex` de LlamaIndex.
*   **Método `_chat_engine`:**
    *   Si `use_context=True`: Crea un `ContextChatEngine`. Configura el retriever para buscar en la base vectorial y aplica post-procesadores (reranking, filtros).
    *   Si `use_context=False`: Crea un `SimpleChatEngine` (chat normal sin documentos).
*   **Integración:** Une el LLM, el modelo de Embeddings y el Vector Store.

#### 📄 `private_gpt/server/ingest/ingest_service.py`
**Importancia:** ⭐⭐⭐⭐
Responsable de "leer" y "aprender" los documentos.
*   Usa `SentenceWindowNodeParser` para dividir los textos en fragmentos inteligentes.
*   Genera embeddings y los guarda en Qdrant.

### 3. Componentes Modulares

#### 📄 `private_gpt/components/llm/llm_component.py`
**Importancia:** ⭐⭐⭐
Abstracción que carga el modelo de lenguaje correcto según `settings.yaml`.
*   Soporta `ollama`, `openai`, `llamacpp`, etc.
*   En el caso de `ollama`, configura el cliente y parámetros como `keep_alive` y `request_timeout`.

#### 📄 `private_gpt/components/vector_store/vector_store_component.py`
**Importancia:** ⭐⭐⭐
Abstracción para la base de datos vectorial.
*   Inicializa el cliente de Qdrant (o Chroma/Milvus).
*   Provee el `retriever` que usa el `ChatService` para buscar información relevante.

---

## 🔄 Flujo de una Petición RAG

1.  **Petición:** Llega a `chat_router.py` con `use_context=True`.
2.  **Servicio:** `ChatService` recibe el mensaje.
3.  **Búsqueda:** El `retriever` (de `VectorStoreComponent`) convierte la pregunta en números (embeddings) y busca fragmentos similares en Qdrant.
4.  **Contexto:** Los fragmentos encontrados se inyectan en el `System Prompt` (definido en `settings.yaml`).
5.  **Generación:** El `LLMComponent` (Ollama) recibe el prompt enriquecido y genera la respuesta.
6.  **Respuesta:** Se envía al usuario vía streaming.
