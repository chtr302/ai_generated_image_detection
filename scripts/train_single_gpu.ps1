param(
    [string]$DataRoot = "data",
    [string]$OutputDir = "outputs/ai_detector",
    [int]$ImageSize = 384,
    [int]$BatchSize = 8,
    [int]$Epochs = 20,
    [int]$GradAccum = 2,
    [int]$Nec = 10,
    [string]$Amp = "fp16",
    [int]$NumWorkers = 0,
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"

# Chay tu root cua repo de import duoc package src.
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$ArgsList = @(
    "-m", "src.model.train",
    "--data-root", $DataRoot,
    "--output-dir", $OutputDir,
    "--image-size", $ImageSize,
    "--batch-size", $BatchSize,
    "--epochs", $Epochs,
    "--grad-accum", $GradAccum,
    "--nec", $Nec,
    "--amp", $Amp,
    "--num-workers", $NumWorkers
)

if ($Resume -ne "") {
    $ArgsList += @("--resume", $Resume)
}

python $ArgsList
