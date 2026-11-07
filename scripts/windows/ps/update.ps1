# Reconstrói a imagem (após mudanças no código) e reinicia a API.
. "$PSScriptRoot\common.ps1"

try {
    Test-Wsl
    Ensure-Docker
    Build-Image

    Write-Step "Reiniciando a API"
    $code = Invoke-Wsl "docker compose up -d api"
    if ($code -ne 0) { throw "docker compose up falhou (código $code)" }

    Wait-Api
    Write-Host "`nAtualização concluída." -ForegroundColor Green
    exit 0
} catch {
    Write-Fail $_.Exception.Message
    exit 1
}
