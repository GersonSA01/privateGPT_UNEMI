# Script para iniciar PrivateGPT en modo GPU
Write-Host "🚀 Iniciando PrivateGPT en modo GPU (CUDA)..." -ForegroundColor Cyan
Write-Host ""

docker-compose --profile ollama-cuda up --build

