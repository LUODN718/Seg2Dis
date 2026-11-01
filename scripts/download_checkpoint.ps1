param(
    [string]$Output = "checkpoints/segdis_rgb_pvtb2_352.pth"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ManifestPath = Join-Path $Root "benchmarks/release.json"
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json

if ($Manifest.status -ne "verified_release" -or -not $Manifest.api_asset_url -or -not $Manifest.sha256) {
    throw "The checkpoint release manifest is incomplete."
}

$Destination = if ([IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $Root $Output }
if (Test-Path -LiteralPath $Destination) {
    $ExistingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Destination).Hash.ToLowerInvariant()
    if ($ExistingHash -eq $Manifest.sha256) {
        Write-Output "Checkpoint already verified: $Destination"
        exit 0
    }
    throw "The destination exists with a different SHA256: $Destination"
}

$CredentialText = "protocol=https`nhost=github.com`n`n" | git credential fill
$Credential = @{}
foreach ($Line in $CredentialText) {
    if ($Line -match "^([^=]+)=(.*)$") { $Credential[$Matches[1]] = $Matches[2] }
}
if (-not $Credential.password) { throw "No GitHub credential is configured for github.com." }

$Parent = Split-Path -Parent $Destination
New-Item -ItemType Directory -Force -Path $Parent | Out-Null
$Temporary = "$Destination.download"
$Headers = @{
    Authorization = "Bearer $($Credential.password)"
    Accept = "application/octet-stream"
    "X-GitHub-Api-Version" = "2022-11-28"
    "User-Agent" = "SegDis-checkpoint-downloader"
}
try {
    Invoke-WebRequest -Uri $Manifest.api_asset_url -Headers $Headers -OutFile $Temporary -MaximumRedirection 5
    $DownloadedHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Temporary).Hash.ToLowerInvariant()
    if ($DownloadedHash -ne $Manifest.sha256) { throw "Downloaded checkpoint SHA256 mismatch." }
    Move-Item -LiteralPath $Temporary -Destination $Destination
    Write-Output "Verified checkpoint: $Destination"
} catch {
    if (Test-Path -LiteralPath $Temporary) { Remove-Item -LiteralPath $Temporary -Force }
    throw
} finally {
    $Credential.Clear()
    $CredentialText = $null
    $Headers.Clear()
}
