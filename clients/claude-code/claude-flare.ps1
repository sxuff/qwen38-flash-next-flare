# Run Claude Code against Flare on the Spark over an SSH tunnel (Windows).
#   claude-flare.ps1 [claude args...]   start Flare if needed, open the tunnel, run claude
#   claude-flare.ps1 -Stop              stop Flare on the Spark
param([switch]$Stop, [Parameter(ValueFromRemainingArguments)] $ClaudeArgs)

# Your setup: set these environment variables, or edit the defaults.
$Spark      = if ($env:FLARE_SPARK) { $env:FLARE_SPARK } else { 'user@spark' }                    # key-based SSH
$SparkLogin = if ($env:FLARE_SPARK_LOGIN) { $env:FLARE_SPARK_LOGIN } else { $Spark }              # a login with PAM limits
$RecipeDir  = if ($env:FLARE_RECIPE_DIR) { $env:FLARE_RECIPE_DIR } else { '~/qwen38-flash-next-flare' }
$LocalPort  = 18000
$Settings   = Join-Path $PSScriptRoot 'settings.json'

if ($Stop) {
    ssh $Spark "bash $RecipeDir/scripts/stop-flare.sh"
    return
}

function Test-FlareRemote {
    ssh -o BatchMode=yes $Spark 'curl -sf http://127.0.0.1:8000/health' *> $null
    return $LASTEXITCODE -eq 0
}

function Test-LocalPort {
    $client = [System.Net.Sockets.TcpClient]::new()
    try { $client.Connect('127.0.0.1', $LocalPort); return $true } catch { return $false } finally { $client.Dispose() }
}

if (-not (Test-FlareRemote)) {
    Write-Host 'Flare is not running. Starting it on the Spark.'
    # The tmux server keeps its memlock limit after Flare stops, so usually no password is needed.
    ssh -o BatchMode=yes $Spark "bash $RecipeDir/scripts/start-flare.sh --no-create"
    if ($LASTEXITCODE -eq 3) {
        Write-Host 'First start since boot: log in once so Flare gets the unlimited memlock limit.'
        ssh -t $SparkLogin "bash $RecipeDir/scripts/start-flare.sh"
    }
    if ($LASTEXITCODE -ne 0) { Write-Error 'Could not start Flare.'; exit 1 }
    Write-Host -NoNewline 'Waiting for Flare to load'
    $deadline = (Get-Date).AddMinutes(4)
    while (-not (Test-FlareRemote)) {
        if ((Get-Date) -gt $deadline) { Write-Error 'Flare did not become healthy; see the run log on the Spark.'; exit 1 }
        Write-Host -NoNewline '.'
        Start-Sleep -Seconds 5
    }
    Write-Host ' ready.'
}

$tunnel = $null
if (-not (Test-LocalPort)) {
    $tunnel = Start-Process ssh -PassThru -WindowStyle Hidden -ArgumentList @(
        '-N', '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30',
        '-L', "${LocalPort}:127.0.0.1:8000", $Spark)
    $deadline = (Get-Date).AddSeconds(20)
    while (-not (Test-LocalPort)) {
        if ($tunnel.HasExited -or (Get-Date) -gt $deadline) { Write-Error 'SSH tunnel failed to open.'; exit 1 }
        Start-Sleep -Milliseconds 300
    }
}

# Variables inherited from another Claude session (for example a desktop app terminal) override the profile; hide them.
$inherited = Get-ChildItem env: | Where-Object { $_.Name -match '^(ANTHROPIC|CLAUDE)' }
$inherited | ForEach-Object { Remove-Item "env:$($_.Name)" }
try {
    claude --settings $Settings @ClaudeArgs
} finally {
    $inherited | ForEach-Object { Set-Item "env:$($_.Name)" $_.Value }
    if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id }
}
