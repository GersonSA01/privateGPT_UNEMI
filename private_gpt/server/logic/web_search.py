import io
import logging
import random
import time
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from llama_index.core.schema import TextNode, NodeWithScore
from pypdf import PdfReader

# Configuración de Logging
logger = logging.getLogger(__name__)

# Constantes de Seguridad
MAX_SEARCH_RESULTS = 1      # Solo el mejor resultado para ser rápidos
MAX_PDF_PAGES = 3           # Leer solo primeras 3 páginas (resumen/intro)
MAX_CONTENT_CHARS = 4000    # Límite estricto de caracteres para no romper el LLM
TIMEOUT_SECONDS = 10        # Tiempo máximo de espera por web

# Headers para parecer un navegador real y no ser bloqueados
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def _clean_html_text(html_content: bytes) -> str:
    """Limpia basura HTML (menús, scripts) y devuelve texto puro."""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Eliminar elementos no deseados
        for trash in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe"]):
            trash.decompose()
            
        # Estrategia de extracción: Buscar el contenedor principal o párrafos
        # Priorizamos etiquetas de contenido
        text_parts = []
        for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li', 'article']):
            text = tag.get_text(strip=True)
            if len(text) > 20: # Filtramos frases muy cortas
                text_parts.append(text)
                
        clean_text = "\n".join(text_parts)
        return clean_text
    except Exception as e:
        logger.warning(f"Error cleaning HTML: {e}")
        return ""

def _extract_pdf_text(pdf_content: bytes) -> str:
    """Lee el binario de un PDF y extrae texto de las primeras páginas."""
    try:
        f = io.BytesIO(pdf_content)
        reader = PdfReader(f)
        text = ""
        # Limitar páginas para velocidad
        limit = min(len(reader.pages), MAX_PDF_PAGES)
        
        for i in range(limit):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text += page_text + "\n"
        
        return text
    except Exception as e:
        logger.warning(f"Error reading PDF: {e}")
        return ""

def search_unemi_universal(query_text: str) -> list[NodeWithScore]:
    """
    Busca con reintentos y backoff exponencial para evitar Rate Limits.
    Descarga el contenido, lo procesa (PDF o HTML) y devuelve un Nodo.
    """
    strict_query = f"{query_text} site:unemi.edu.ec"
    logger.info(f"🌍 WEB AGENT: Searching for '{strict_query}'...")
    
    nodes = []
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Espera aleatoria antes de buscar (Evita parecer un bot agresivo)
            if attempt > 0:
                wait_time = random.uniform(1, 3)
                logger.info(f"⏳ Waiting {wait_time:.1f}s before retry {attempt+1}...")
                time.sleep(wait_time)
            else:
                # Primera vez: espera corta aleatoria
                time.sleep(random.uniform(0.5, 1.5))
            
            # 1. Ejecutar Búsqueda con backend 'html' (más robusto contra bloqueos)
            with DDGS() as ddgs:
                results = list(ddgs.text(strict_query, max_results=MAX_SEARCH_RESULTS, backend="html"))
                
            if not results:
                logger.info("🤷 WEB AGENT: No results found.")
                return []

            # Si llegamos aquí, tuvimos éxito. Procesamos el primer resultado.
            r = results[0]
            url = r['href']
            title = r['title']
            snippet = r['body']
            
            logger.info(f"📥 WEB AGENT: Downloading {url}...")
            
            try:
                # 2. Descargar Contenido
                response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
                response.raise_for_status()
                
                content_type = response.headers.get('Content-Type', '').lower()
                extracted_text = ""
                source_type = "UNKNOWN"

                # 3. Detección y Parsing
                if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                    source_type = "PDF DOCUMENT"
                    extracted_text = _extract_pdf_text(response.content)
                else:
                    source_type = "WEBSITE"
                    extracted_text = _clean_html_text(response.content)

                # Fallback: Si falló la extracción, usamos el snippet del buscador
                if not extracted_text or len(extracted_text) < 50:
                    extracted_text = f"(Content extraction failed. Using summary): {snippet}"

                # 4. Truncado de Seguridad
                if len(extracted_text) > MAX_CONTENT_CHARS:
                    extracted_text = extracted_text[:MAX_CONTENT_CHARS] + "\n...[TRUNCATED]..."

                # 5. Formateo para el LLM
                final_content = (
                    f"--- WEB SEARCH RESULT ({source_type}) ---\n"
                    f"TITLE: {title}\n"
                    f"URL: {url}\n"
                    f"CONTENT:\n{extracted_text}\n"
                    f"--- END WEB RESULT ---"
                )
                
                # Creamos el nodo con un score "artificial" medio
                # para que el LLM sepa que es información externa
                node = TextNode(text=final_content)
                node.metadata = {"url": url, "title": title, "source": "web_search"}
                
                nodes.append(NodeWithScore(node=node, score=0.75))
                
                # Si todo sale bien, rompemos el bucle de reintentos
                logger.info(f"✅ WEB AGENT: Successfully processed result on attempt {attempt+1}")
                break
                
            except Exception as e:
                logger.error(f"❌ Error processing URL {url}: {e}")
                # Si falla el procesamiento de la URL, no reintentamos la búsqueda
                break

        except Exception as e:
            error_str = str(e)
            logger.warning(f"⚠️ Search attempt {attempt+1}/{max_retries} failed: {e}")
            
            # Manejo específico de rate limits
            if "ratelimit" in error_str.lower() or "rate limit" in error_str.lower() or "429" in error_str:
                wait_time = (attempt + 1) * 5  # Espera 5, 10, 15 segundos
                logger.info(f"⏳ Rate limit detected. Cooling down for {wait_time} seconds...")
                if attempt < max_retries - 1:  # No esperar en el último intento
                    time.sleep(wait_time)
            elif attempt < max_retries - 1:
                # Para otros errores, espera corta antes de reintentar
                wait_time = (attempt + 1) * 2
                logger.info(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # Último intento falló
                logger.error(f"❌ Critical Search Error after {max_retries} attempts: {e}")
                import traceback
                traceback.print_exc()
                return []

    logger.info(f"✅ WEB AGENT: Returning {len(nodes)} nodes")
    return nodes
