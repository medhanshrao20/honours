param(
    [ValidateSet('all','audit','classify','preprocess','features','prune','rank','validate','cross-modal','loso','control')]
    [string]$Stage = 'all',
    [ValidateSet('paper','senior')][string]$Profile = 'paper',
    [switch]$Smoke,
    [switch]$FromRaw,
    [switch]$UseReferenceResults,
    [int]$Jobs = 1
)
$ErrorActionPreference = 'Stop'
$Python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    py -3.11 -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required.' }
}
& $Python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
$RunArgs = @((Join-Path $PSScriptRoot 'main.py'),'--stage',$Stage,'--profile',$Profile,'--jobs',"$Jobs")
if ($Smoke) { $RunArgs += '--smoke' }
if ($FromRaw) { $RunArgs += '--from-raw' }
if ($UseReferenceResults) { $RunArgs += '--use-reference-results' }
& $Python @RunArgs
if ($LASTEXITCODE -ne 0) { throw 'Pipeline failed; see the error above.' }
