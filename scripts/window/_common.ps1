Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-RepoRoot {
  return (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
}

function Get-DateStamp {
  return Get-Date -Format 'yyyy-MM-dd-HHmm.ss'
}

function Ensure-Directory {
  param([string]$Path)

  New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function New-FileLinkOrCopy {
  param(
    [string]$SourcePath,
    [string]$DestinationPath
  )

  $destinationDir = Split-Path -Parent $DestinationPath
  if ($destinationDir) {
    Ensure-Directory $destinationDir
  }

  if (Test-Path $DestinationPath) {
    Remove-Item $DestinationPath -Force
  }

  try {
    New-Item -ItemType HardLink -Path $DestinationPath -Value $SourcePath | Out-Null
  } catch {
    Copy-Item $SourcePath $DestinationPath -Force
  }
}

function Expand-GzipFile {
  param(
    [string]$SourcePath,
    [string]$DestinationPath
  )

  $destinationDir = Split-Path -Parent $DestinationPath
  if ($destinationDir) {
    Ensure-Directory $destinationDir
  }

  $inputStream = [System.IO.File]::OpenRead($SourcePath)
  try {
    $gzipStream = New-Object System.IO.Compression.GzipStream(
      $inputStream,
      [System.IO.Compression.CompressionMode]::Decompress
    )
    try {
      $outputStream = [System.IO.File]::Create($DestinationPath)
      try {
        $gzipStream.CopyTo($outputStream)
      } finally {
        $outputStream.Dispose()
      }
    } finally {
      $gzipStream.Dispose()
    }
  } finally {
    $inputStream.Dispose()
  }
}

function Get-ExistingFile {
  param([string[]]$Paths)

  foreach ($path in $Paths) {
    if (Test-Path $path) {
      return $path
    }
  }

  return $null
}

function Get-PythonExecutable {
  $repoRoot = Get-RepoRoot
  $venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
  if (Test-Path $venvPython) {
    return $venvPython
  }

  return 'python'
}

function python {
  & (Get-PythonExecutable) @args
}
