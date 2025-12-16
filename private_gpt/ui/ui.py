"""This file should be imported if and only if you want to run the UI locally."""

import base64
import logging
import time
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.core.types import TokenGen
from pydantic import BaseModel

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.recipes.summarize.summarize_service import SummarizeService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import logo_svg

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "private_gpt/ui/avatar-bot.ico"
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "avatar-bot.ico"

UI_TAB_TITLE = "My Private GPT"

SOURCES_SEPARATOR = "<hr>Sources: \n"


class Modes(str, Enum):
    RAG_MODE = "RAG"
    SEARCH_MODE = "Search"
    BASIC_CHAT_MODE = "Basic"
    SUMMARIZE_MODE = "Summarize"


MODES: list[Modes] = [
    Modes.RAG_MODE,
    Modes.SEARCH_MODE,
    Modes.BASIC_CHAT_MODE,
    Modes.SUMMARIZE_MODE,
]


class Source(BaseModel):
    file: str
    page: str
    text: str

    class Config:
        frozen = True

    @staticmethod
    def curate_sources(sources: list[Chunk]) -> list["Source"]:
        curated_sources = []

        for chunk in sources:
            doc_metadata = chunk.document.doc_metadata

            file_name = doc_metadata.get("file_name", "-") if doc_metadata else "-"
            page_label = doc_metadata.get("page_label", "-") if doc_metadata else "-"

            source = Source(file=file_name, page=page_label, text=chunk.text)
            curated_sources.append(source)
            curated_sources = list(
                dict.fromkeys(curated_sources).keys()
            )  # Unique sources only

        return curated_sources


@singleton
class PrivateGptUi:
    @inject
    def __init__(
        self,
        ingest_service: IngestService,
        chat_service: ChatService,
        chunks_service: ChunksService,
        summarizeService: SummarizeService,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service
        self._summarize_service = summarizeService

        # Cache the UI blocks
        self._ui_block = None

        self._selected_filename = None

        # Initialize system prompt based on default mode
        default_mode_map = {mode.value: mode for mode in Modes}
        self._default_mode = default_mode_map.get(
            settings().ui.default_mode, Modes.RAG_MODE
        )
        self._system_prompt = self._get_default_system_prompt(self._default_mode)

    def _chat(
        self, message: str, history: list[list[str]], mode: Modes, *_: Any
    ) -> Any:
        def yield_deltas(completion_gen: CompletionGen) -> Iterable[str]:
            full_response: str = ""
            stream = completion_gen.response
            for delta in stream:
                if isinstance(delta, str):
                    full_response += str(delta)
                elif isinstance(delta, ChatResponse):
                    full_response += delta.delta or ""
                yield full_response
                time.sleep(0.02)

            if completion_gen.sources:
                full_response += SOURCES_SEPARATOR
                cur_sources = Source.curate_sources(completion_gen.sources)
                sources_text = "\n\n\n"
                used_files = set()
                for index, source in enumerate(cur_sources, start=1):
                    if f"{source.file}-{source.page}" not in used_files:
                        sources_text = (
                            sources_text
                            + f"{index}. {source.file} (page {source.page}) \n\n"
                        )
                        used_files.add(f"{source.file}-{source.page}")
                sources_text += "<hr>\n\n"
                full_response += sources_text
            yield full_response

        def yield_tokens(token_gen: TokenGen) -> Iterable[str]:
            full_response: str = ""
            for token in token_gen:
                full_response += str(token)
                yield full_response

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = []

            for interaction in history:
                history_messages.append(
                    ChatMessage(content=interaction[0], role=MessageRole.USER)
                )
                if len(interaction) > 1 and interaction[1] is not None:
                    history_messages.append(
                        ChatMessage(
                            # Remove from history content the Sources information
                            content=interaction[1].split(SOURCES_SEPARATOR)[0],
                            role=MessageRole.ASSISTANT,
                        )
                    )

            # max 20 messages to try to avoid context overflow
            return history_messages[:20]

        new_message = ChatMessage(content=message, role=MessageRole.USER)
        all_messages = [*build_history(), new_message]
        # If a system prompt is set, add it as a system message
        if self._system_prompt:
            all_messages.insert(
                0,
                ChatMessage(
                    content=self._system_prompt,
                    role=MessageRole.SYSTEM,
                ),
            )
        match mode:
            case Modes.RAG_MODE:
                # Use only the selected file for the query
                context_filter = None
                if self._selected_filename is not None:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested():
                        if (
                            ingested_document.doc_metadata["file_name"]
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                query_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                    context_filter=context_filter,
                )
                yield from yield_deltas(query_stream)
            case Modes.BASIC_CHAT_MODE:
                llm_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=False,
                )
                yield from yield_deltas(llm_stream)

            case Modes.SEARCH_MODE:
                response = self._chunks_service.retrieve_relevant(
                    text=message, limit=4, prev_next_chunks=0
                )

                sources = Source.curate_sources(response)

                yield "\n\n\n".join(
                    f"{index}. **{source.file} "
                    f"(page {source.page})**\n "
                    f"{source.text}"
                    for index, source in enumerate(sources, start=1)
                )
            case Modes.SUMMARIZE_MODE:
                # Summarize the given message, optionally using selected files
                context_filter = None
                if self._selected_filename:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested():
                        if (
                            ingested_document.doc_metadata["file_name"]
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                summary_stream = self._summarize_service.stream_summarize(
                    use_context=True,
                    context_filter=context_filter,
                    instructions=message,
                )
                yield from yield_tokens(summary_stream)

    # On initialization and on mode change, this function set the system prompt
    # to the default prompt based on the mode (and user settings).
    @staticmethod
    def _get_default_system_prompt(mode: Modes) -> str:
        p = ""
        match mode:
            # For query chat mode, obtain default system prompt from settings
            case Modes.RAG_MODE:
                p = settings().ui.default_query_system_prompt
            # For chat mode, obtain default system prompt from settings
            case Modes.BASIC_CHAT_MODE:
                p = settings().ui.default_chat_system_prompt
            # For summarization mode, obtain default system prompt from settings
            case Modes.SUMMARIZE_MODE:
                p = settings().ui.default_summarization_system_prompt
            # For any other mode, clear the system prompt
            case _:
                p = ""
        return p

    @staticmethod
    def _get_default_mode_explanation(mode: Modes) -> str:
        match mode:
            case Modes.RAG_MODE:
                return "Get contextualized answers from selected files."
            case Modes.SEARCH_MODE:
                return "Find relevant chunks of text in selected files."
            case Modes.BASIC_CHAT_MODE:
                return "Chat with the LLM using its training data. Files are ignored."
            case Modes.SUMMARIZE_MODE:
                return "Generate a summary of the selected files. Prompt to customize the result."
            case _:
                return ""

    def _set_system_prompt(self, system_prompt_input: str) -> None:
        logger.info(f"Setting system prompt to: {system_prompt_input}")
        self._system_prompt = system_prompt_input

    def _set_explanatation_mode(self, explanation_mode: str) -> None:
        self._explanation_mode = explanation_mode

    def _set_current_mode(self, mode: Modes) -> Any:
        self.mode = mode
        self._set_system_prompt(self._get_default_system_prompt(mode))
        self._set_explanatation_mode(self._get_default_mode_explanation(mode))
        interactive = self._system_prompt is not None
        return [
            gr.update(placeholder=self._system_prompt, interactive=interactive),
            gr.update(value=self._explanation_mode),
        ]

    # -------------------------------------------------------------------------
    # 🔍 MODIFICACIÓN: LISTAR ARCHIVOS CON FILTRO DE ROL (MULTI-ROL)
    # -------------------------------------------------------------------------
    def _list_ingested_files(self, role_filter: str | None = "VER_TODOS") -> list[list[str]]:
        """Return a list of file names filtered by role.
        If role_filter is "VER_TODOS" all files are returned.
        """
        if not role_filter:
            role_filter = "VER_TODOS"
        files = set()
        all_docs = self._ingest_service.list_ingested()
        for ingested_document in all_docs:
            if ingested_document.doc_metadata is None:
                continue
            file_name = ingested_document.doc_metadata.get("file_name", "[FILE NAME MISSING]")
            doc_roles = ingested_document.doc_metadata.get("role", ["general"])
            if isinstance(doc_roles, str):
                doc_roles = [doc_roles]
            if role_filter == "VER_TODOS" or role_filter in doc_roles:
                files.add(file_name)
        logger.info(f"✅ Archivos visibles en lista: {len(files)}")
        return [[row] for row in files]
    # 📤 MODIFICACIÓN: SUBIR ARCHIVO CON MÚLTIPLES ROLES
    # -------------------------------------------------------------------------
    def _upload_file(self, files: list[str], roles: list[str]) -> None:
        # Si el usuario no selecciona nada, forzamos 'general'
        if not roles or len(roles) == 0:
            roles = ["general"]
            
        logger.debug("Loading count=%s files with roles=%s", len(files), roles)
        paths = [Path(file) for file in files]

        # remove all existing Documents with name identical to a new file upload:
        file_names = [path.name for path in paths]
        doc_ids_to_delete = []
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"] in file_names
            ):
                doc_ids_to_delete.append(ingested_document.doc_id)
        if len(doc_ids_to_delete) > 0:
            logger.info(
                "Uploading file(s) which were already ingested: %s document(s) will be replaced.",
                len(doc_ids_to_delete),
            )
            for doc_id in doc_ids_to_delete:
                self._ingest_service.delete(doc_id)

        # Pasamos la lista de roles (plural)
        self._ingest_service.bulk_ingest([(str(path.name), path) for path in paths], role_tag=roles)

    def _delete_all_files(self) -> None:
        ingested_files = self._ingest_service.list_ingested()
        logger.debug("Deleting count=%s files", len(ingested_files))
        for ingested_document in ingested_files:
            self._ingest_service.delete(ingested_document.doc_id)

    def _delete_selected_file(self) -> None:
        logger.debug("Deleting selected %s", self._selected_filename)
        # Note: keep looping for pdf's (each page became a Document)
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"]
                == self._selected_filename
            ):
                self._ingest_service.delete(ingested_document.doc_id)

    def _get_file_roles(self, filename: str) -> list[str]:
        """Return current role metadata for a given filename."""
        docs = self._ingest_service.list_ingested()
        for doc in docs:
            if doc.doc_metadata and doc.doc_metadata.get("file_name") == filename:
                roles = doc.doc_metadata.get("role", ["general"])
                if isinstance(roles, list):
                    return roles
                return [roles]
        return ["general"]

    def _deselect_selected_file(self) -> list[Any]:
        self._selected_filename = None
        return [
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value="All files"),
            gr.update(visible=False),
            gr.update(value=["general"]),
            gr.update(value="### 🛠️ Editar Documento Seleccionado"),
        ]

    def _selected_a_file(self, select_data: gr.SelectData) -> list[Any]:
        self._selected_filename = select_data.value
        current_roles = self._get_file_roles(self._selected_filename)
        return [
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value=self._selected_filename),
            gr.update(visible=True),
            gr.update(value=current_roles),
            gr.update(value=f"### 🛠️ Editar Roles: {self._selected_filename}"),
        ]

    # -------------------------------------------------------------------------
    # 🎭 FUNCIONES PARA CONTROL DE "MODAL" DE SUBIDA
    # -------------------------------------------------------------------------
    def _toggle_upload_modal(self) -> list[Any]:
        """Muestra el panel de subida y oculta la lista."""
        return [
            gr.update(visible=True),   # upload_group (Modal)
            gr.update(visible=False),  # list_group (Tabla)
            gr.update(visible=False),  # filter_group (Filtros)
        ]

    def _close_upload_modal(self) -> list[Any]:
        """Oculta el panel de subida y muestra la lista."""
        return [
            gr.update(visible=False),  # upload_group
            gr.update(visible=True),   # list_group
            gr.update(visible=True),   # filter_group
            None,                      # Limpiar archivo seleccionado
            ["general"]                # Resetear roles a default
        ]

    def _upload_and_close(self, files: list[str] | None, roles: list[str], current_filter: str) -> list[Any]:
        """Sube el archivo y cierra el modal automáticamente."""
        if not files:
            # No hacer nada si no hay archivo
            return [
                gr.update(),  # No cambiar visibilidad del modal
                gr.update(),  # No cambiar visibilidad de la lista
                gr.update(),  # No cambiar visibilidad de filtros
                gr.update(),  # No actualizar dataset
                gr.update(),  # No limpiar input
                gr.update(),  # No resetear roles
            ]
            
        # 1. Ejecutar lógica de subida
        self._upload_file(files, roles)
        
        # 2. Refrescar la lista (usando el filtro actual)
        new_list = self._list_ingested_files(current_filter)
        
        # 3. Retornar actualizaciones de UI (Cerrar modal, mostrar lista actualizada)
        return [
            gr.update(visible=False),  # Ocultar modal
            gr.update(visible=True),   # Mostrar lista
            gr.update(visible=True),   # Mostrar filtros
            new_list,                  # Actualizar datos de la lista
            None,                      # Limpiar input de archivo
            ["general"]                # Resetear checkboxes
        ]

    def _update_selected_roles(
        self, new_roles: list[str], current_filter: str
    ) -> list[Any]:
        if not self._selected_filename:
            return [
                gr.update(value="⚠️ Selecciona un documento antes de guardar."),
                gr.update(),
            ]

        if not new_roles:
            new_roles = ["general"]

        self._ingest_service.update_file_roles(self._selected_filename, new_roles)

        current_filter = current_filter or "VER_TODOS"
        return [
            gr.update(value=f"✅ Roles actualizados para: {self._selected_filename}"),
            self._list_ingested_files(current_filter),
        ]

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(primary_hue=slate),
            css=".logo { "
            "display:flex;"
            "background-color: #C7BAFF;"
            "height: 80px;"
            "border-radius: 8px;"
            "align-content: center;"
            "justify-content: center;"
            "align-items: center;"
            "}"
            ".logo img { height: 25% }"
            ".contain { display: flex !important; flex-direction: column !important; }"
            "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
            "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
            "#col { height: calc(100vh - 112px - 16px) !important; }"
            "hr { margin-top: 1em; margin-bottom: 1em; border: 0; border-top: 1px solid #FFF; }"
            ".avatar-image { background-color: antiquewhite; border-radius: 2px; }"
            ".footer { text-align: center; margin-top: 20px; font-size: 14px; display: flex; align-items: center; justify-content: center; }"
            ".footer-zylon-link { display:flex; margin-left: 5px; text-decoration: auto; color: var(--body-text-color); }"
            ".footer-zylon-link:hover { color: #C7BAFF; }"
            ".footer-zylon-ico { height: 20px; margin-left: 5px; background-color: antiquewhite; border-radius: 2px; }",
        ) as blocks:
            with gr.Row():
                gr.HTML(f"<div class='logo'/><img src={logo_svg} alt=PrivateGPT></div")

            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    default_mode = self._default_mode
                    mode = gr.Radio(
                        [mode.value for mode in MODES],
                        label="Mode",
                        value=default_mode,
                    )
                    explanation_mode = gr.Textbox(
                        placeholder=self._get_default_mode_explanation(default_mode),
                        show_label=False,
                        max_lines=3,
                        interactive=False,
                    )
                    
                    # =========================================================
                    # 1. BOTÓN PRINCIPAL Y FILTROS (VISTA NORMAL)
                    # =========================================================
                    with gr.Group() as filter_group:
                        with gr.Row():
                            new_doc_btn = gr.Button(
                                "Nuevo Documento", 
                                variant="primary", 
                                size="sm"
                            )
                            filter_selector = gr.Dropdown(
                                choices=[
                                    "VER_TODOS",
                                    "general", 
                                    "es_estudiante", 
                                    "es_profesor", 
                                    "es_administrativo",
                                    "es_externo",
                                    "es_inscripcionaspirante",
                                    "es_inscripcionpostulante",
                                    "es_postulante",
                                    "es_postulanteempleo",
                                    "es_inscripcionadmision"
                                ],
                                value="VER_TODOS",
                                label="Filtrar Archivos",
                                container=False,
                                scale=2
                            )

                    # =========================================================
                    # 2. "MODAL" DE SUBIDA (OCULTO POR DEFECTO)
                    # =========================================================
                    with gr.Group(visible=False) as upload_group:
                        gr.Markdown("### 📤 Subir Nuevos Documentos")
                        
                        # Área de Drag & Drop
                        file_drop = gr.File(
                            label="Arrastra tus archivos aquí (PDF, TXT, DOCX, MD, CSV...)",
                            file_count="multiple",
                            type="filepath",
                            height=150,
                            file_types=[
                                ".pdf",
                                ".txt",
                                ".md",
                                ".docx",
                                ".csv",
                                ".json",
                            ],
                        )
                        
                        # Checkbox de Roles
                        roles_checkbox = gr.CheckboxGroup(
                            choices=[
                                "general", 
                                "es_estudiante", 
                                "es_profesor", 
                                "es_administrativo",
                                "es_externo",
                                "es_inscripcionaspirante",
                                "es_inscripcionpostulante",
                                "es_postulante",
                                "es_postulanteempleo",
                                "es_inscripcionadmision"
                            ],
                            value=["general"],
                            label="¿Quién puede ver estos documentos?",
                            info="Selecciona uno o varios roles."
                        )
                        
                        with gr.Row():
                            cancel_btn = gr.Button("Cancelar", variant="secondary")
                            confirm_btn = gr.Button("✅ Subir y Procesar", variant="primary")

                    # =========================================================
                    # 3. LISTA DE ARCHIVOS (VISTA NORMAL)
                    # =========================================================
                    with gr.Group() as list_group:
                        ingested_dataset = gr.List(
                            self._list_ingested_files,
                            headers=["File name"],
                            label="Documentos Ingestados",
                            height=300,
                            interactive=False,
                            render=False,
                        )
                        ingested_dataset.render()
                        
                        deselect_file_button = gr.components.Button(
                            "De-select selected file", size="sm", interactive=False
                        )
                        selected_text = gr.components.Textbox(
                            "All files", label="Selected for Query or Deletion", max_lines=1
                        )
                        delete_file_button = gr.components.Button(
                            "🗑️ Delete selected file",
                            size="sm",
                            visible=settings().ui.delete_file_button_enabled,
                            interactive=False,
                        )
                        delete_files_button = gr.components.Button(
                            "⚠️ Delete ALL files",
                            size="sm",
                            visible=settings().ui.delete_all_files_button_enabled,
                        )

                    # =========================================================
                    # 4. PANEL DE EDICIÓN DE ROLES
                    # =========================================================
                    with gr.Group(visible=False) as edit_role_group:
                        edit_title = gr.Markdown("### 🛠️ Editar Documento Seleccionado")
                        with gr.Row():
                            edit_roles_checkbox = gr.CheckboxGroup(
                                choices=[
                                    "general",
                                    "es_estudiante",
                                    "es_profesor",
                                    "es_administrativo",
                                    "es_externo",
                                    "es_inscripcionaspirante",
                                    "es_inscripcionpostulante",
                                    "es_postulante",
                                    "es_postulanteempleo",
                                    "es_inscripcionadmision",
                                ],
                                label="Asignar Roles",
                                value=["general"],
                            )
                            save_roles_btn = gr.Button(
                                "💾 Guardar Cambios", variant="primary", scale=0
                            )

                    # =========================================================
                    # 5. CONEXIÓN DE EVENTOS (WIRING)
                    # =========================================================
                    
                    # A. ABRIR MODAL
                    new_doc_btn.click(
                        self._toggle_upload_modal,
                        outputs=[upload_group, list_group, filter_group]
                    )
                    
                    # B. CANCELAR MODAL
                    cancel_btn.click(
                        self._close_upload_modal,
                        outputs=[upload_group, list_group, filter_group, file_drop, roles_checkbox]
                    )
                    
                    # C. CONFIRMAR SUBIDA
                    confirm_btn.click(
                        self._upload_and_close,
                        inputs=[file_drop, roles_checkbox, filter_selector],
                        outputs=[upload_group, list_group, filter_group, ingested_dataset, file_drop, roles_checkbox]
                    )

                    # D. FILTRADO DE LISTA
                    filter_selector.change(
                        fn=self._list_ingested_files,
                        inputs=[filter_selector],
                        outputs=[ingested_dataset]
                    )

                    # E. EVENTOS DE SELECCIÓN Y BORRADO
                    deselect_file_button.click(
                        self._deselect_selected_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                            edit_role_group,
                            edit_roles_checkbox,
                            edit_title,
                        ],
                    )
                    
                    ingested_dataset.select(
                        fn=self._selected_a_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                            edit_role_group,
                            edit_roles_checkbox,
                            edit_title,
                        ],
                    )

                    save_roles_btn.click(
                        self._update_selected_roles,
                        inputs=[edit_roles_checkbox, filter_selector],
                        outputs=[edit_title, ingested_dataset],
                    )
                    
                    delete_file_button.click(
                        self._delete_selected_file,
                    ).then(
                        self._deselect_selected_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                            edit_role_group,
                            edit_roles_checkbox,
                            edit_title,
                        ],
                    ).then(
                        self._list_ingested_files,
                        inputs=[filter_selector],
                        outputs=[ingested_dataset]
                    )
                    
                    delete_files_button.click(
                        self._delete_all_files,
                    ).then(
                        self._deselect_selected_file,
                        outputs=[
                            delete_file_button,
                            deselect_file_button,
                            selected_text,
                            edit_role_group,
                            edit_roles_checkbox,
                            edit_title,
                        ],
                    ).then(
                        self._list_ingested_files,
                        inputs=[filter_selector],
                        outputs=[ingested_dataset]
                    )

                    system_prompt_input = gr.Textbox(
                        placeholder=self._system_prompt,
                        label="System Prompt",
                        lines=2,
                        interactive=True,
                        render=False,
                    )
                    # When mode changes, set default system prompt, and other stuffs
                    mode.change(
                        self._set_current_mode,
                        inputs=mode,
                        outputs=[system_prompt_input, explanation_mode],
                    )
                    # On blur, set system prompt to use in queries
                    system_prompt_input.blur(
                        self._set_system_prompt,
                        inputs=system_prompt_input,
                    )

                    def get_model_label() -> str | None:
                        """Get model label from llm mode setting YAML.

                        Raises:
                            ValueError: If an invalid 'llm_mode' is encountered.

                        Returns:
                            str: The corresponding model label.
                        """
                        # Get model label from llm mode setting YAML
                        # Labels: local, openai, openailike, sagemaker, mock, ollama
                        config_settings = settings()
                        if config_settings is None:
                            raise ValueError("Settings are not configured.")

                        # Get llm_mode from settings
                        llm_mode = config_settings.llm.mode

                        # Mapping of 'llm_mode' to corresponding model labels
                        model_mapping = {
                            "llamacpp": config_settings.llamacpp.llm_hf_model_file,
                            "openai": config_settings.openai.model,
                            "openailike": config_settings.openai.model,
                            "azopenai": config_settings.azopenai.llm_model,
                            "sagemaker": config_settings.sagemaker.llm_endpoint_name,
                            "mock": llm_mode,
                            "ollama": config_settings.ollama.llm_model,
                            "gemini": config_settings.gemini.model,
                        }

                        if llm_mode not in model_mapping:
                            print(f"Invalid 'llm mode': {llm_mode}")
                            return None

                        return model_mapping[llm_mode]

                with gr.Column(scale=7, elem_id="col"):
                    # Determine the model label based on the value of PGPT_PROFILES
                    model_label = get_model_label()
                    if model_label is not None:
                        label_text = (
                            f"LLM: {settings().llm.mode} | Model: {model_label}"
                        )
                    else:
                        label_text = f"LLM: {settings().llm.mode}"

                    _ = gr.ChatInterface(
                        self._chat,
                        chatbot=gr.Chatbot(
                            label=label_text,
                            show_copy_button=True,
                            elem_id="chatbot",
                            render=False,
                            avatar_images=(
                                None,
                                AVATAR_BOT,
                            ),
                        ),
                        additional_inputs=[mode, system_prompt_input],
                    )

            with gr.Row():
                avatar_byte = AVATAR_BOT.read_bytes()
                f_base64 = f"data:image/png;base64,{base64.b64encode(avatar_byte).decode('utf-8')}"
                gr.HTML(
                    f"<div class='footer'><a class='footer-zylon-link' href='https://zylon.ai/'>Maintained by Zylon <img class='footer-zylon-ico' src='{f_base64}' alt=Zylon></a></div>"
                )

        return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the gradio UI, at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path, favicon_path=AVATAR_BOT)


if __name__ == "__main__":
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)