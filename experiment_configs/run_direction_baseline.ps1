$repo = "C:\Users\lixianfeng\Desktop\JointSCOD"
$py = "C:\Users\lixianfeng\anaconda3\envs\JointSCOD\python.exe"
if (-not (Test-Path -LiteralPath $py)) { $py = "python" }

$SodPath = "$repo\SOD\SOD-TrainDataset\DUTS-TR"
$CodPath = "$repo\COD\TrainDataset"
$Backbone = "$repo\pvt_v2_b2.pth"
$OutDir = "$repo\output_direction_baseline"

& $py -u "$repo\train_joint_direction.py" `
  --experiment baseline `
  --sod_path $SodPath `
  --cod_path $CodPath `
  --backbone_checkpoint $Backbone `
  --output $OutDir `
  --batch_size 8 `
  --epochs 50 `
  --base_lr 1e-4 `
  --gpu 0 `
  --seed 42

if ($LASTEXITCODE -ne 0) {
  Write-Host "TRAIN FAILED: baseline exit=$LASTEXITCODE" -ForegroundColor Red
  exit $LASTEXITCODE
}

$ckpt = "$OutDir\best.pth"
if (-not (Test-Path -LiteralPath $ckpt)) {
  Write-Host "Missing checkpoint: $ckpt" -ForegroundColor Red
  exit 1
}

$SodTestRoot = "$repo\SOD\SOD-TestDataset"
$CodTestRoot = "$repo\COD\TestDataset"

foreach ($d in @("CAMO", "CHAMELEON", "COD10K", "NC4K")) {
  $out = "$repo\results_direction\baseline\COD\$d"
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  & $py -u "$repo\test.py" `
    --variant v3_joint_shared `
    --path "$CodTestRoot\$d" `
    --checkpoint $ckpt `
    --output $out `
    --gpu 0
}

foreach ($d in @("DUTS-TE", "ECSSD", "HKU-IS", "DUT-OMRON", "PASCAL-S")) {
  $out = "$repo\results_direction\baseline\SOD\$d"
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  & $py -u "$repo\test.py" `
    --variant v3_joint_shared `
    --path "$SodTestRoot\$d" `
    --checkpoint $ckpt `
    --output $out `
    --gpu 0
}
