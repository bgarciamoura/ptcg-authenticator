# Diagnóstico rápido: WSL, Docker, containers e saúde da API.
. "$PSScriptRoot\common.ps1"

try {
    Write-Step "WSL"
    Test-Wsl
    Write-Ok "distro $($Config.Distro) encontrada"

    Write-Step "Docker"
    if (-not (Test-DockerRunning)) { Write-Warn "Docker parado (iniciar.bat liga)"; exit 0 }
    Write-Ok "rodando"

    Write-Step "Containers do projeto"
    $null = Invoke-Wsl "docker compose ps"

    Write-Step "API (porta $($Config.ApiPort))"
    $status = Get-ApiStatus
    switch ($status) {
        "ok"      { Write-Ok "pronta: http://localhost:$($Config.ApiPort)" }
        "loading" { Write-Warn "carregando modelos" }
        "error"   { Write-Fail "falha ao carregar modelos (ver-logs.bat)" }
        default   { Write-Warn "não responde" }
    }
    exit 0
} catch {
    Write-Fail $_.Exception.Message
    exit 1
}
