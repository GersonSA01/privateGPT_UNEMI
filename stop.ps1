# Script para detener PrivateGPT
param(
    [string]$Mode = "all"
)

Write-Host "🛑 Deteniendo PrivateGPT..." -ForegroundColor Yellow

if ($Mode -eq "cpu") {
    docker-compose --profile ollama-cpu down
}
elseif ($Mode -eq "gpu") {
    docker-compose --profile ollama-cuda down
}
else {
    docker-compose --profile ollama-cpu down
    docker-compose --profile ollama-cuda down
    Write-Host "Todos los servicios han sido detenidos" -ForegroundColor Green
}

