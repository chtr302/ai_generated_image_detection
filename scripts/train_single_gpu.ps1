param(
    [string]$DataRoot = "",
    [string]$HfDataset = "Rajarshi-Roy-research/Defactify_Image_Dataset",
    [string]$HfCacheDir = "",
    [switch]$HfNoStreaming,
    [int]$HfShuffleBuffer = 10000,
    [string]$OutputDir = "outputs/ai_detector",
    [int]$ImageSize = 384,
    [int]$BatchSize = 8,
    [int]$Epochs = 20,
    [int]$GradAccum = 2,
    [int]$Nec = 10,
    [string]$Amp = "fp16",
    [int]$NumWorkers = 0,
    [string]$Resume = "",
    [int]$MaxTrainSteps = 0
)

$ErrorActionPreference = "Stop"

# Run from repo root so Python can import src.* modules.
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$ArgsList = @(
    "-m", "src.model.train",
    "--output-dir", $OutputDir,
    "--image-size", $ImageSize,
    "--batch-size", $BatchSize,
    "--epochs", $Epochs,
    "--grad-accum", $GradAccum,
    "--nec", $Nec,
    "--amp", $Amp,
    "--num-workers", $NumWorkers
)

if ($DataRoot -ne "") {
    $ArgsList += @("--data-root", $DataRoot)
} else {
    $ArgsList += @("--hf-dataset", $HfDataset, "--hf-shuffle-buffer", $HfShuffleBuffer)
    if ($HfCacheDir -ne "") {
        $ArgsList += @("--hf-cache-dir", $HfCacheDir)
    }
    if ($HfNoStreaming) {
        $ArgsList += "--hf-no-streaming"
    }
}

if ($Resume -ne "") {
    $ArgsList += @("--resume", $Resume)
}
if ($MaxTrainSteps -gt 0) {
    $ArgsList += @("--max-train-steps", $MaxTrainSteps)
}

python $ArgsList
