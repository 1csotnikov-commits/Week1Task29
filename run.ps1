# Запуск приложения «Локальный LLM-чат» (День 27).
# Использование:
#   powershell -ExecutionPolicy Bypass -File .\run.ps1
#   или просто:  .\run.ps1
# Сервер поднимется на http://127.0.0.1:8000 (страница откроется в браузере).

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$ollamaUrl = "http://localhost:11434"
$host = "127.0.0.1"
$port = 8000

Write-Host "Проверка доступности Ollama ($ollamaUrl) ..." -ForegroundColor Cyan
try {
    Invoke-RestMethod -Uri "$ollamaUrl/api/version" -TimeoutSec 3 | Out-Null
    Write-Host "Ollama доступна." -ForegroundColor Green
} catch {
    Write-Host "ВНИМАНИЕ: Ollama недоступна." -ForegroundColor Yellow
    Write-Host "Запустите её командой:  C:\LLM\Ollama\ollama.exe serve" -ForegroundColor Yellow
}

Write-Host "Запуск веб-сервера: http://$($host):$port" -ForegroundColor Cyan
python -m uvicorn app.main:app --reload --host $host --port $port
