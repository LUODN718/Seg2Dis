param(
 [Parameter(Mandatory=$true)][string]$Python,
 [Parameter(Mandatory=$true)][string]$Config,
 [Parameter(Mandatory=$true)][string]$BackboneCheckpoint,
 [Parameter(Mandatory=$true)][string]$Output,
 [string]$UscRoot,
 [string]$SodTrain,
 [string]$CodTrain,
 [string]$Device='cuda:0',
 [switch]$Preflight
)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$arguments=@((Join-Path $root 'train.py'),'--config',$Config,'--backbone_checkpoint',$BackboneCheckpoint,'--output',$Output,'--device',$Device)
foreach($item in @(@('usc_root',$UscRoot),@('sod_train',$SodTrain),@('cod_train',$CodTrain))){
 if($item[1]){$arguments+=@('--'+$item[0],$item[1])}
}
if($Preflight){$arguments+='--preflight'}
& $Python @arguments
if($LASTEXITCODE -ne 0){throw "SegDis exited with code $LASTEXITCODE"}
