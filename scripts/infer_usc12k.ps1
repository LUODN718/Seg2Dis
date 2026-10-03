param(
    [Parameter(Mandatory=$true)][string]$UscRoot,
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$OutputRoot = ".\predictions\segdis",
    [ValidateSet("label", "color", "separate", "all")][string]$Format = "color",
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
        --image_size 352 `
        --gpu $Gpu
    if ($LASTEXITCODE -ne 0) { throw "Inference failed: $Split" }
}
