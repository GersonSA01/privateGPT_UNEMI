import sys
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# CONFIGURACIÓN
# ¡IMPORTANTE! Cambia esto según tu modelo de embeddings.
# OpenAI = 1536, Nomic = 768, Local default = 1024 o 384.
VECTOR_SIZE = 768  
COLLECTION_NAME = "make_this_parameterizable_per_api_call" # Nombre por defecto de PrivateGPT

def resetear():
    try:
        print("🔌 Conectando a Qdrant...")
        client = QdrantClient(url="http://localhost:6333")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    # 1. BORRAR SI EXISTE
    if client.collection_exists(COLLECTION_NAME):
        print(f"🗑️  Borrando colección '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
    else:
        print(f"⚠️ La colección '{COLLECTION_NAME}' no existía.")

    # 2. RECREAR VACÍA
    print(f"✨ Creando colección '{COLLECTION_NAME}' vacía (Dim: {VECTOR_SIZE})...")
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    print("✅ ¡Listo! Base de datos reiniciada. No necesitas reiniciar Docker.")

if __name__ == "__main__":
    resetear()