import sys
from qdrant_client import QdrantClient
import json

# 1. CONEXIÓN
# Si estás ejecutando esto desde TU MÁQUINA (fuera de Docker), usa localhost.
# Si Qdrant está en otro puerto, cámbialo aquí.
try:
    print("🔌 Conectando a Qdrant en localhost:6333...")
    client = QdrantClient(url="http://localhost:6333")
    collections = client.get_collections()
except Exception as e:
    print(f"❌ Error conectando: {e}")
    print("   Intenta ejecutar este script DENTRO del contenedor de docker si no tienes el puerto expuesto.")
    sys.exit(1)

# 2. SELECCIONAR COLECCIÓN
if not collections.collections:
    print("⚠️ No hay colecciones en la base de datos (está vacía).")
    sys.exit(0)

col_name = collections.collections[0].name
print(f"📂 Colección encontrada: '{col_name}'")

# 3. LEER CHUNKS (Scroll)
# Pedimos 100 puntos para ver qué hay
response, _ = client.scroll(
    collection_name=col_name,
    limit=100,
    with_payload=True,
    with_vectors=False
)

print(f"\n🔍 INSPECCIONANDO LOS PRIMEROS {len(response)} CHUNKS:\n")
print("-" * 60)

found_pages = []

for point in response:
    payload = point.payload
    
    # Intentamos extraer metadatos típicos de LlamaIndex/PrivateGPT
    file_name = payload.get("file_name", "Desconocido")
    # A veces es 'page_label', a veces está dentro de 'metadata'
    page = payload.get("page_label", None)
    
    if not page and "metadata" in payload:
        page = payload["metadata"].get("page_label", "N/A")
    
    # Extraer un poco de texto para validar
    content = payload.get("_node_content", "")
    text_preview = "Texto no legible"
    if content:
        try:
            # LlamaIndex guarda el contenido como JSON string
            content_json = json.loads(content)
            text_preview = content_json.get("text", "")[:50].replace("\n", " ")
        except:
            text_preview = str(content)[:50]

    print(f"📄 Archivo: {file_name}")
    print(f"   📍 Página: {page}")
    print(f"   📝 Inicio texto: {text_preview}...")
    print("-" * 60)
    
    if page:
        found_pages.append(page)

print("\n📊 RESUMEN ESTADÍSTICO:")
print(f"Total chunks leídos: {len(response)}")
print(f"Páginas encontradas: {list(set(found_pages))}")

if "1" in str(found_pages) and "2" not in str(found_pages) and "10" not in str(found_pages):
    print("\n🚨 DIAGNÓSTICO: ALERTA ROJA 🚨")
    print("Solo veo referencias a la PÁGINA 1 (o muy pocas).")
    print("Esto confirma que Ollama cortó el archivo al ingerirlo porque el 'chunk_size' era muy grande.")
elif len(found_pages) > 5:
    print("\n✅ DIAGNÓSTICO: BUENO")
    print("Veo múltiples páginas diferentes. La ingesta parece correcta.")