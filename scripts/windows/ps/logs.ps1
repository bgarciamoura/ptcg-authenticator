# Mostra os logs da API ao vivo (Ctrl+C para sair).
. "$PSScriptRoot\common.ps1"

try {
    Test-Wsl
    if (-not (Test-DockerRunning)) { throw "Docker está parado. Rode iniciar.bat primeiro." }
    Write-Step "Logs da API (Ctrl+C para sair)"
    $null = Invoke-Wsl "docker compose logs --tail 100 -f api"
    exit 0
} catch {
    Write-Fail $_.Exception.Message
    exit 1
}
