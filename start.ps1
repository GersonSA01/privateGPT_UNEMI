# Script principal para iniciar PrivateGPT
# Lee el modo desde .env y ejecuta el perfil correspondiente

param(
    [string]$Mode = ""
)

# Si no se especifica modo, leer desde .env
if ([string]::IsNullOrEmpty($Mode)) {
    if (Test-Path .env) {
        $envContent = Get-Content .env -Raw
        if ($envContent -match 'OLLAMA_MODE=(cpu|gpu)') {
            $Mode = $matches[1]
        }
        else {
            $Mode = "cpu"
        }
    }
    else {
        $Mode = "cpu"
    }
}

Write-Host "📋 Modo detectado: $Mode" -ForegroundColor Cyan
Write-Host ""

if ($Mode -eq "gpu" -or $Mode -eq "cuda") {
    Write-Host "🚀 Iniciando PrivateGPT en modo GPU (CUDA)..." -ForegroundColor Cyan
    docker-compose --profile ollama-cuda up --build
}
else {
    Write-Host "🚀 Iniciando PrivateGPT en modo CPU..." -ForegroundColor Cyan
    docker-compose --profile ollama-cpu up --build
}

