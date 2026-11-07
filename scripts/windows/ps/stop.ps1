# Para o projeto (o Docker continua rodando).
. "$PSScriptRoot\common.ps1"

try {
    Test-Wsl
    if (-not (Test-DockerRunning)) { Write-Ok "Docker já está parado; nada a fazer"; exit 0 }
    Write-Step "Parando o projeto"
    $code = Invoke-Wsl "docker compose stop"
    if ($code -ne 0) { throw "docker compose stop falhou (código $code)" }
    Write-Ok "projeto parado"
    exit 0
} catch {
    Write-Fail $_.Exception.Message
    exit 1
}
