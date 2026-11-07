# Configuração dos scripts Windows.
#
# Quando esta pasta é aberta direto do WSL (\\wsl.localhost\<distro>\...\scripts\windows),
# distro e caminho do projeto são deduzidos automaticamente. Se a pasta foi
# copiada para a Área de Trabalho, valem os padrões abaixo — ajuste se preciso.

$Config = @{
    Distro           = "Ubuntu-24.04"
    ProjectPath      = "/home/bgarciamoura/projects/pokemon-card-authenticator"
    ApiPort          = 8088
    DockerTimeoutSec = 90
    # Espera da API já rodando (o build é feito antes, com saída visível)
    ApiTimeoutSec    = 300
}

# Dedução automática a partir de onde os scripts estão: \\wsl.localhost\Ubuntu-24.04\home\...\scripts\windows\ps
$here = $PSScriptRoot
if ($here -match '^\\\\(wsl\.localhost|wsl\$)\\([^\\]+)\\(.+)\\scripts\\windows\\ps\\?$') {
    $Config.Distro      = $Matches[2]
    $Config.ProjectPath = "/" + ($Matches[3] -replace '\\', '/')
}

# Porta: se houver .env no projeto com API_PORT, ele vence
try {
    $envFile = Join-Path $here "..\..\..\.env"
    if (Test-Path $envFile) {
        $line = Get-Content $envFile | Where-Object { $_ -match '^\s*API_PORT\s*=\s*(\d+)' } | Select-Object -First 1
        if ($line -and $Matches[1]) { $Config.ApiPort = [int]$Matches[1] }
    }
} catch {}
