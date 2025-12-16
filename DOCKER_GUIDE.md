# Guía de uso de PrivateGPT con Docker - CPU/GPU Switch

Esta guía te ayudará a cambiar fácilmente entre modo CPU y GPU para ejecutar PrivateGPT con Ollama.

## 🎯 Configuración Rápida

### Opción 1: Usar scripts helper (Recomendado)

#### Para CPU (desarrollo local):
```powershell
.\start-cpu.ps1
```

#### Para GPU (servidor con NVIDIA):
```powershell
.\start-gpu.ps1
```

#### Para usar el modo configurado en .env:
```powershell
.\start.ps1
```

#### Para detener:
```powershell
.\stop.ps1
```

### Opción 2: Comandos Docker Compose directos

#### CPU:
```powershell
docker-compose --profile ollama-cpu up
```

#### GPU:
```powershell
docker-compose --profile ollama-cuda up
```

## 📝 Configuración mediante archivo .env

1. Edita el archivo `.env`:
```powershell
# Para CPU (desarrollo)
OLLAMA_MODE=cpu

# Para GPU (servidor)
OLLAMA_MODE=gpu
```

2. Luego ejecuta:
```powershell
.\start.ps1
```

## 🔄 Cambiar entre CPU y GPU

### Método 1: Editar .env
1. Edita `.env` y cambia `OLLAMA_MODE` a `cpu` o `gpu`
2. Ejecuta `.\start.ps1`

### Método 2: Usar scripts específicos
- `.\start-cpu.ps1` - Inicia en modo CPU
- `.\start-gpu.ps1` - Inicia en modo GPU

### Método 3: Especificar modo en línea de comandos
```powershell
.\start.ps1 -Mode cpu
.\start.ps1 -Mode gpu
```

## 📦 Requisitos

### Para CPU:
- Docker Desktop instalado
- Al menos 8GB de RAM recomendado

### Para GPU:
- Docker Desktop con soporte NVIDIA Container Toolkit
- GPU NVIDIA con CUDA
- NVIDIA Container Toolkit instalado

## 🚀 Primera ejecución

En la primera ejecución, Docker:
1. Construirá las imágenes necesarias (puede tardar varios minutos)
2. Descargará la imagen de Ollama
3. Iniciará los servicios

**IMPORTANTE**: Después de iniciar, necesitas descargar los modelos de Ollama:

```powershell
# Para CPU
docker-compose exec ollama-cpu ollama pull llama3.1-instruct-q4_K_M 
docker-compose exec ollama-cpu ollama pull nomic-embed-text

# Para GPU
docker-compose exec ollama-cuda ollama pull llama3.1-instruct-q4_K_M 
docker-compose exec ollama-cuda ollama pull nomic-embed-text
```

## 🌐 Acceso a los servicios

Una vez iniciado:
- **PrivateGPT API y UI**: http://localhost:8001
- **Ollama API**: http://localhost:11434

## 🔍 Ver logs

```powershell
# CPU
docker-compose --profile ollama-cpu logs -f

# GPU
docker-compose --profile ollama-cuda logs -f
```

## 🛑 Detener servicios

```powershell
# Detener CPU
docker-compose --profile ollama-cpu down

# Detener GPU
docker-compose --profile ollama-cuda down

# O usar el script
.\stop.ps1
```

## 💡 Tips

- Los modelos se almacenan en `./models` (compartido entre CPU y GPU)
- Los datos de PrivateGPT están en `./local_data`
- Si cambias entre CPU y GPU, no necesitas descargar los modelos de nuevo
- Usa `docker-compose down -v` si quieres eliminar los volúmenes también

## ⚠️ Troubleshooting

### Error: "No se puede conectar a Ollama"
- Asegúrate de que el servicio de Ollama esté corriendo
- Verifica los logs: `docker-compose logs ollama-cpu` o `docker-compose logs ollama-cuda`

### Error con GPU: "Cannot connect to the Docker daemon"
- Asegúrate de tener NVIDIA Container Toolkit instalado
- Verifica: `docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi`

### Error: "Port already in use"
- Detén otros servicios que usen los puertos 8001 o 11434
- O cambia los puertos en `docker-compose.yaml`

