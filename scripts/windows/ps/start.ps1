# Inicia tudo: WSL -> Docker -> (build na 1ª vez) -> projeto -> navegador.
param([switch] $NoBrowser)
. "$PSScriptRoot\common.ps1"

try {
    Write-Host "Verificador de Cartas Pokémon — inicialização" -ForegroundColor White
    Write-Host "projeto: $($Config.Distro):$($Config.ProjectPath)" -ForegroundColor DarkGray

    Write-Step "Verificando o WSL"
    Test-Wsl
    Write-Ok "distro $($Config.Distro) encontrada"

    Ensure-Docker

    if (-not (Test-ImageBuilt)) {
        Write-Warn "primeira execução: a imagem ainda não existe"
        Build-Image
    }

    Write-Step "Subindo o projeto (docker compose up -d api)"
    $code = Invoke-Wsl "docker compose up -d api"
    if ($code -ne 0) { throw "docker compose falhou (código $code). Veja a saída acima." }
    Write-Ok "container iniciado"

    Wait-Api
    if (-not $NoBrowser) { Open-Browser }

    Write-Host "`nTudo pronto. Pode fechar esta janela; a API continua rodando." -ForegroundColor Green
    exit 0
} catch {
    Write-Fail $_.Exception.Message
    exit 1
}
