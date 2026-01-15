from typing import Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.server.utils.auth import authenticated

ingest_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])

class IngestTextBody(BaseModel):
    file_name: str = Field(examples=["Avatar: The Last Airbender"])
    text: str = Field(examples=["Avatar is set in an Asian..."])

# --- NUEVO MODELO DE METADATOS (CORREGIDO) ---
class UpdateMetadataBody(BaseModel):
    roles: list[str] = Field(default=["general"], description="Roles permitidos")
    valid_from: Optional[str] = Field(default=None, description="Fecha inicio YYYY-MM-DD")
    valid_to: Optional[str] = Field(default=None, description="Fecha fin YYYY-MM-DD")
    is_infinite: bool = Field(default=True, description="Ignorar fechas")
    
    # === AGREGADO AQUÍ ===
    db_id: Optional[int] = Field(default=None, description="ID numérico de la Base de Datos Django")
    access_url: Optional[str] = Field(default=None, description="URL pública/media del archivo")
    faq_answer: Optional[str] = Field(default=None, description="La respuesta literal de la FAQ")
    # =====================

class IngestResponse(BaseModel):
    object: Literal["list"]
    model: Literal["private-gpt"]
    data: list[IngestedDoc]

@ingest_router.post("/ingest/file", tags=["Ingestion"])
def ingest_file(
    request: Request, 
    file: UploadFile, 
    db_id: Optional[int] = None  # <--- NUEVO PARÁMETRO OPCIONAL
) -> IngestResponse:
    service = request.state.injector.get(IngestService)
    if file.filename is None:
        raise HTTPException(400, "No file name provided")
    
    # Pasamos el db_id directamente al servicio
    ingested_documents = service.ingest_bin_data(
        file.filename, 
        file.file, 
        db_id=db_id
    )
    
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)

@ingest_router.post("/ingest/text", tags=["Ingestion"])
def ingest_text(request: Request, body: IngestTextBody) -> IngestResponse:
    service = request.state.injector.get(IngestService)
    if len(body.file_name) == 0:
        raise HTTPException(400, "No file name provided")
    ingested_documents = service.ingest_text(body.file_name, body.text)
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)

@ingest_router.get("/ingest/list", tags=["Ingestion"])
def list_ingested(request: Request) -> IngestResponse:
    service = request.state.injector.get(IngestService)
    ingested_documents = service.list_ingested()
    return IngestResponse(object="list", model="private-gpt", data=ingested_documents)

@ingest_router.delete("/ingest/{doc_id}", tags=["Ingestion"])
def delete_ingested(request: Request, doc_id: str) -> None:
    service = request.state.injector.get(IngestService)
    service.delete(doc_id)

# --- ENDPOINT METADATA (CONECTADO) ---
@ingest_router.post("/ingest/{doc_id}/metadata", tags=["Ingestion"])
def update_doc_metadata(request: Request, doc_id: str, body: UpdateMetadataBody) -> None:
    """Update roles, validity dates AND DB references of an ingested document."""
    service = request.state.injector.get(IngestService)
    
    # Llamamos a la función en el servicio pasando los nuevos campos
    service.update_doc_metadata(
        doc_id, 
        body.roles, 
        body.valid_from, 
        body.valid_to, 
        body.is_infinite,
        # === PASAMOS LOS NUEVOS DATOS AL SERVICIO ===
        db_id=body.db_id,
        access_url=body.access_url,
        faq_answer=body.faq_answer
    )