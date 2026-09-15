param(
    [Parameter(Mandatory=$true)][string]$UscRoot,
    [Parameter(Mandatory=$true)][string]$BackboneCheckpoint,
    [string]$Output = ".\outputs\segdis_b4",
    [int]$Gpu = 0
)

$ErrorActionPreference = "Stop"
python -u .\train.py `
    --usc_root $UscRoot `
    --backbone_checkpoint $BackboneCheckpoint `
    --output $Output `
    --variant B4 `
    --epochs 50 `
    --batch_size 8 `
    --base_lr 1e-4 `
    --image_size 352 `
    --type_weight 0.5 `
    --calibration_weight 0.5 `
    --workers 4 `
    --seed 42 `
    --gpu $Gpu
if ($LASTEXITCODE -ne 0) { throw "SegDis training failed with exit code $LASTEXITCODE" }
