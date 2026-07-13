param(
    [string]$DataRoot = "data",
    [string]$OutputDir = "outputs/ai_detector",
    [int]$ImageSize = 384,
    [int]$BatchSize = 8,
    [int]$Epochs = 20,
    [int]$GradAccum = 2,
    [int]$Nec = 10,
    [string]$Amp = "fp16",
    [int]$NumWorkers = 4,
    [string]$Resume = ""
)

$ErrorActionPreference = "Stop"

# Script nay phu hop server co torchrun va 2 GPU CUDA.
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$ArgsList = @(
    "--nproc_per_node=2",
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

torchrun $ArgsList
