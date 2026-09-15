param(
    [Parameter(Mandatory=$true)][string]$UscRoot,
    [Parameter(Mandatory=$true)][string]$BackboneCheckpoint,
    [string]$OutputRoot = ".\outputs\ablation",
    [string]$ResultRoot = ".\results\ablation",
    [int]$Gpu = 0
)

$ErrorActionPreference = "Stop"
foreach ($Variant in @("B0", "B1", "B2", "B3", "B4")) {
    $Output = "$OutputRoot\$Variant"
    python -u .\train.py `
        --usc_root $UscRoot `
        --backbone_checkpoint $BackboneCheckpoint `
        --output $Output `
        --variant $Variant `
        --epochs 50 `
        --batch_size 8 `
        --base_lr 1e-4 `
        --image_size 352 `
        --type_weight 0.5 `
        --calibration_weight 0.5 `
        --workers 4 `
        --seed 42 `
        --gpu $Gpu
    if ($LASTEXITCODE -ne 0) { throw "Training failed: $Variant" }

    foreach ($Split in @("Scene-A", "Scene-B", "Scene-C", "Scene-D", "val")) {
        python -u .\evaluate.py `
            --usc_root $UscRoot `
            --checkpoint "$Output\best.pth" `
            --split $Split `
            --output "$ResultRoot\$Variant\$Split.json" `
            --gpu $Gpu
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $Variant / $Split" }
    }
}
