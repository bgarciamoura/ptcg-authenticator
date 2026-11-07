# Funções compartilhadas pelos scripts. Carregado com: . "$PSScriptRoot\common.ps1"
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

. "$PSScriptRoot\config.ps1"

# ------------------------------------------------------------------ saída
function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    [!] $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "`n[ERRO] $msg" -ForegroundColor Red }

# ------------------------------------------------------------------- WSL
function Test-Wsl {
    <# Verifica se o WSL existe e se a distro configurada está instalada. #>
    $wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if (-not $wsl) { throw "WSL não está instalado. Instale com: wsl --install" }
    # a saída do wsl -l vem em UTF-16 com caracteres nulos; limpa antes de comparar
    $distros = (& wsl.exe -l -q 2>$null) -join "`n" -replace "`0", "" -split "`r?`n" |
        ForEach-Object { $_.Trim() } | Where-Object { $_ }
    if ($distros -notcontains $Config.Distro) {
        throw "Distro '$($Config.Distro)' não encontrada. Distros disponíveis: $($distros -join ', '). Ajuste ps\config.ps1."
    }
}

function Invoke-Wsl {
    <#
      Executa um comando bash dentro da distro, já no diretório do projeto.
      A saída vai direto ao console (Out-Host); só o código de saída volta.
    #>
    param(
        [Parameter(Mandatory)] [string] $Command,
        [switch] $AsRoot,
        [switch] $Quiet
    )
    $args = @("-d", $Config.Distro)
    if ($AsRoot) { $args += @("-u", "root") }
    $args += @("--cd", $Config.ProjectPath, "--", "bash", "-lc", $Command)
    if ($Quiet) { & wsl.exe @args *> $null } else { & wsl.exe @args | Out-Host }
    return $LASTEXITCODE
}

# ---------------------------------------------------------------- Docker
function Test-DockerRunning {
    return (Invoke-Wsl "docker info" -Quiet) -eq 0
}

function Start-DockerService {
    <# Inicia o serviço do Docker dentro do WSL (systemd ou SysV) como root. #>
    Write-Step "Iniciando o serviço do Docker no WSL ($($Config.Distro))"
    $code = Invoke-Wsl -AsRoot -Quiet "systemctl start docker 2>/dev/null || service docker start"
    if ($code -ne 0) { Write-Warn "o comando de start devolveu código $code; aguardando mesmo assim" }

    $deadline = (Get-Date).AddSeconds($Config.DockerTimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-DockerRunning) { Write-Host ""; Write-Ok "Docker respondendo"; return }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 3
    }
    throw "Docker não respondeu em $($Config.DockerTimeoutSec)s. Abra o WSL e rode: sudo systemctl status docker"
}

function Ensure-Docker {
    Write-Step "Verificando o Docker"
    if (Test-DockerRunning) { Write-Ok "Docker já está rodando"; return }
    Write-Warn "Docker parado"
    Start-DockerService
}

function Test-ImageBuilt {
    return (Invoke-Wsl "docker image inspect card-authenticator:latest" -Quiet) -eq 0
}

function Build-Image {
    Write-Step "Construindo a imagem (primeira vez: 10-30 min, baixa vários GB)"
    Write-Host "    O progresso aparece abaixo. Não feche esta janela." -ForegroundColor DarkGray
    $code = Invoke-Wsl "docker compose build api"
    if ($code -ne 0) { throw "o build falhou (código $code). Confira a conexão e rode de novo: as etapas prontas ficam em cache." }
    Write-Ok "imagem construída"
}

# ------------------------------------------------------------------- API
function Get-ApiStatus {
    <# Retorna 'ok', 'loading', 'error' ou 'down'. #>
    $url = "http://127.0.0.1:$($Config.ApiPort)/health"
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5
        if ($r.StatusCode -eq 200) { return "ok" }
        return "loading"
    } catch {
        $resp = $_.Exception.Response
        if ($resp -and [int]$resp.StatusCode -eq 503) {
            try {
                $body = (New-Object IO.StreamReader($resp.GetResponseStream())).ReadToEnd() | ConvertFrom-Json
                if ($body.status -eq "error") { return "error" }
            } catch {}
            return "loading"
        }
        return "down"
    }
}

function Wait-Api {
    <# Espera a API responder 200 em /health, mostrando o progresso. #>
    Write-Step "Aguardando a API ficar pronta (porta $($Config.ApiPort))"
    $deadline = (Get-Date).AddSeconds($Config.ApiTimeoutSec)
    $last = ""
    while ((Get-Date) -lt $deadline) {
        $status = Get-ApiStatus
        if ($status -eq "ok") { Write-Host ""; Write-Ok "API pronta"; return }
        if ($status -eq "error") {
            throw "a API iniciou mas falhou ao carregar os modelos. Veja os logs (ver-logs.bat)."
        }
        if ($status -ne $last) { Write-Host "`n    estado: $status " -NoNewline; $last = $status }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 5
    }
    throw "a API não ficou pronta em $($Config.ApiTimeoutSec)s. Veja os logs (ver-logs.bat)."
}

function Open-Browser {
    $url = "http://localhost:$($Config.ApiPort)"
    Write-Step "Abrindo $url"
    Start-Process $url
}
