param(
    [Parameter(Mandatory=$true)][string]$UscRoot,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$OutputRoot = ".\results\segdis_b4",
    [int]$Gpu = 0
)

$ErrorActionPreference = "Stop"
foreach ($Split in @("Scene-A", "Scene-B", "Scene-C", "Scene-D", "val")) {
    python -u .\evaluate.py `
        --usc_root $UscRoot `
        --checkpoint $Checkpoint `
        --split $Split `
        --output "$OutputRoot\$Split.json" `
        --gpu $Gpu
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $Split" }
}
