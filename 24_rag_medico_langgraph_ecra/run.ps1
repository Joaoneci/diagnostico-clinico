$ErrorActionPreference = 'Stop'
$ScriptDir = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir '..\..')).Path
$EnvFile = Join-Path $RepoRoot '.env'

if (-not (Test-Path -LiteralPath $EnvFile)) {
    Write-Warning "Crie $EnvFile com GOOGLE_API_KEY."
}

Set-Location $ScriptDir

# Stop other exercises just to free up ports and memory if needed
# This is a good practice seen in other exercises
. (Join-Path (Split-Path $ScriptDir -Parent) 'lib_docker_exercicios.ps1')
Stop-OtherExerciseDocker -CurrentExerciseDirectory $ScriptDir

docker compose up --build -d

Write-Host "Docker: http://localhost:8502"
Write-Host 'Parar: docker compose down'
Write-Host 'Logs:  docker compose logs -f'
