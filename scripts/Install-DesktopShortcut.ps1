param(
    [string]$DataDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "data"),
    [string]$DesktopDir = [Environment]::GetFolderPath("Desktop")
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $repoRoot "Launch-Polarcam.cmd"
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project-local Python is missing. Follow README.md or Polarcam User Manual.docx before installing the shortcut."
}
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Launcher is missing: $launcher"
}
if (-not [IO.Path]::IsPathRooted($DataDir)) {
    $DataDir = Join-Path $repoRoot $DataDir
}
$DataDir = [IO.Path]::GetFullPath($DataDir)
if ($DataDir -match '["%\r\n]') {
    throw "The data directory cannot contain quotes, percent signs, or newlines."
}
if (-not (Test-Path -LiteralPath $DesktopDir -PathType Container)) {
    throw "Desktop directory does not exist: $DesktopDir"
}
$shortcutPath = Join-Path $DesktopDir "Polarcam Lab.lnk"
if (Test-Path -LiteralPath $shortcutPath) {
    throw "Shortcut already exists: $shortcutPath. Remove it explicitly before recreating it."
}
New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $env:ComSpec
$shortcut.Arguments = '/d /c ""{0}" --data-dir "{1}""' -f $launcher, $DataDir
$shortcut.WorkingDirectory = $repoRoot
$shortcut.IconLocation = "$python,0"
$shortcut.Description = "Polarcam testing application; data: $DataDir"
$shortcut.Save()
Write-Output "Created: $shortcutPath"
Write-Output "Data directory: $DataDir"
