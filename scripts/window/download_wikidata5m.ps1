. (Join-Path $PSScriptRoot '_common.ps1')

$repoRoot = Get-RepoRoot
$baseDir = Join-Path $repoRoot 'data/wikidata5m'
Ensure-Directory $baseDir

Write-Host "working directory: $repoRoot"

$downloads = @(
  @('wikidata5m_text.txt.gz', 'https://huggingface.co/datasets/intfloat/wikidata5m/resolve/main/wikidata5m_text.txt.gz'),
  @('wikidata5m_transductive.tar.gz', 'https://huggingface.co/datasets/intfloat/wikidata5m/resolve/main/wikidata5m_transductive.tar.gz'),
  @('wikidata5m_inductive.tar.gz', 'https://huggingface.co/datasets/intfloat/wikidata5m/resolve/main/wikidata5m_inductive.tar.gz'),
  @('wikidata5m_alias.tar.gz', 'https://huggingface.co/datasets/intfloat/wikidata5m/resolve/main/wikidata5m_alias.tar.gz')
)

foreach ($download in $downloads) {
  $outputPath = Join-Path $baseDir $download[0]
  Invoke-WebRequest -UseBasicParsing -Uri $download[1] -OutFile $outputPath
}

Push-Location $baseDir
try {
  tar -xzf 'wikidata5m_transductive.tar.gz'
  tar -xzf 'wikidata5m_inductive.tar.gz'
  tar -xzf 'wikidata5m_alias.tar.gz'
  Expand-GzipFile -SourcePath (Join-Path $baseDir 'wikidata5m_text.txt.gz') -DestinationPath (Join-Path $baseDir 'wikidata5m_text.txt')
} finally {
  Pop-Location
}

$transDir = Join-Path $repoRoot 'data/wiki5m_trans'
Ensure-Directory $transDir
New-FileLinkOrCopy -SourcePath (Join-Path $baseDir 'wikidata5m_relation.txt') -DestinationPath (Join-Path $transDir 'wikidata5m_relation.txt')
New-FileLinkOrCopy -SourcePath (Join-Path $baseDir 'wikidata5m_text.txt') -DestinationPath (Join-Path $transDir 'wikidata5m_text.txt')
New-FileLinkOrCopy -SourcePath (Join-Path $baseDir 'wikidata5m_entity.txt') -DestinationPath (Join-Path $transDir 'wikidata5m_entity.txt')
New-FileLinkOrCopy -SourcePath (Join-Path $baseDir 'wikidata5m_transductive_train.txt') -DestinationPath (Join-Path $transDir 'train.txt')
New-FileLinkOrCopy -SourcePath (Join-Path $baseDir 'wikidata5m_transductive_valid.txt') -DestinationPath (Join-Path $transDir 'valid.txt')
New-FileLinkOrCopy -SourcePath (Join-Path $baseDir 'wikidata5m_transductive_test.txt') -DestinationPath (Join-Path $transDir 'test.txt')

$indDir = Join-Path $repoRoot 'data/wiki5m_ind'
Ensure-Directory $indDir
New-FileLinkOrCopy -SourcePath (Join-Path $baseDir 'wikidata5m_relation.txt') -DestinationPath (Join-Path $indDir 'wikidata5m_relation.txt')
New-FileLinkOrCopy -SourcePath (Join-Path $baseDir 'wikidata5m_text.txt') -DestinationPath (Join-Path $indDir 'wikidata5m_text.txt')
New-FileLinkOrCopy -SourcePath (Join-Path $baseDir 'wikidata5m_entity.txt') -DestinationPath (Join-Path $indDir 'wikidata5m_entity.txt')
New-FileLinkOrCopy -SourcePath (Join-Path $baseDir 'wikidata5m_inductive_train.txt') -DestinationPath (Join-Path $indDir 'train.txt')
New-FileLinkOrCopy -SourcePath (Join-Path $baseDir 'wikidata5m_inductive_valid.txt') -DestinationPath (Join-Path $indDir 'valid.txt')
New-FileLinkOrCopy -SourcePath (Join-Path $baseDir 'wikidata5m_inductive_test.txt') -DestinationPath (Join-Path $indDir 'test.txt')

Write-Host 'Done'
