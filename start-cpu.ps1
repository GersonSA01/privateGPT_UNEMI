# Script para iniciar PrivateGPT en modo CPU
Write-Host "Iniciando PrivateGPT en modo CPU..." -ForegroundColor Cyan
Write-Host ""

docker-compose --profile ollama-cpu up --build

