import sys
from qdrant_client import QdrantClient

def borrar_todo():
    # 1. CONEXIÓN
    try:
        print("🔌 Conectando a Qdrant en localhost:6333...")
        client = QdrantClient(url="http://localhost:6333")
    except Exception as e:
        print(f"❌ Error conectando: {e}")
        sys.exit(1)

    # 2. LISTAR COLECCIONES
    collections_response = client.get_collections()
    
    if not collections_response.collections:
        print("✅ La base de datos ya está vacía. No hay nada que borrar.")
        return

    print(f"\n⚠️  ¡ATENCIÓN! Se han encontrado {len(collections_response.collections)} colecciones.")
    for col in collections_response.collections:
        print(f"   - {col.name}")

    # 3. CONFIRMACIÓN DE SEGURIDAD
    print("\n" + "!"*40)
    print("ESTA ACCIÓN ES IRREVERSIBLE.")
    print("Se eliminarán TODOS los datos y documentos indexados.")
    print("!"*40)
    
    confirmacion = input("\nPara confirmar, escribe 'BORRAR' (en mayúsculas) y presiona Enter: ")

    if confirmacion != "BORRAR":
        print("❌ Operación cancelada. No se ha borrado nada.")
        sys.exit(0)

    # 4. BORRADO
    print("\n🗑️  Iniciando borrado...")
    
    for col in collections_response.collections:
        try:
            client.delete_collection(collection_name=col.name)
            print(f"   ✅ Colección '{col.name}' eliminada correctamente.")
        except Exception as e:
            print(f"   ❌ Error eliminando '{col.name}': {e}")

    print("\n✨ Limpieza completada. Qdrant está vacío.")

if __name__ == "__main__":
    borrar_todo()