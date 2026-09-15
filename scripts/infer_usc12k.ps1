param(
    [Parameter(Mandatory=$true)][string]$UscRoot,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$OutputRoot = ".\predictions\segdis_b4",
    [ValidateSet("label", "color", "separate", "all")][string]$Format = "label",
    [int]$Gpu = 0
)

$ErrorActionPreference = "Stop"
foreach ($Split in @("Scene-A", "Scene-B", "Scene-C", "Scene-D")) {
    python -u .\infer.py `
        --usc_root $UscRoot `
        --checkpoint $Checkpoint `
        --split $Split `
        --output "$OutputRoot\$Split" `
        --format $Format `
        --gpu $Gpu
    if ($LASTEXITCODE -ne 0) { throw "Inference failed: $Split" }
}
