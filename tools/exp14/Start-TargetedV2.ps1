[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Python,
    [Parameter(Mandatory = $true)][string] $RunRoot,
    [ValidateRange(1,8)][int] $Workers = 4,
    [ValidateSet('A6','A7','A7B','A8')][string] $Phase = 'A6'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$base = [Environment]::GetEnvironmentVariable('BORESCOPE_IMAGE_BASE_URL', 'User')
$model = [Environment]::GetEnvironmentVariable('BORESCOPE_IMAGE_MODEL', 'User')
if ($base -cne 'https://direct-api.ccccapi.cc' -or $model -cne 'gpt-image-2.5-sunburst') {
    throw 'Windows relay settings differ from the frozen protocol.'
}

$keyFile = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'BorescopeImageRelay\relay_key.dpapi'
$keyHash = (Get-FileHash -LiteralPath $keyFile -Algorithm SHA256).Hash
$secureKey = ConvertTo-SecureString -String (Get-Content -LiteralPath $keyFile -TotalCount 1)
$bstr = [IntPtr]::Zero
$process = $null
try {
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = (Resolve-Path -LiteralPath $Python).Path
    $startInfo.WorkingDirectory = $repo
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $runPath = (Resolve-Path -LiteralPath $RunRoot).Path
    $module = switch ($Phase) {
        'A8' { 'tools.exp14.final_v4' }
        'A7B' { 'tools.exp14.targeted_v3_remaining' }
        'A7' { 'tools.exp14.targeted_v3' }
        'A6' { 'tools.exp14.targeted_v2' }
    }
    foreach ($argument in @('-u', '-X', 'utf8', '-m', $module, 'live',
                            '--root', $runPath, '--workers', "$Workers")) {
        $startInfo.ArgumentList.Add($argument)
    }
    $startInfo.Environment['EXP14_IMAGE_API_KEY'] = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    try {
        $process = [Diagnostics.Process]::Start($startInfo)
    } finally {
        [void] $startInfo.Environment.Remove('EXP14_IMAGE_API_KEY')
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        $bstr = [IntPtr]::Zero
    }
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw "$Phase runner stopped. Inspect private logs and retain request locks." }
    if ((Get-FileHash -LiteralPath $keyFile -Algorithm SHA256).Hash -cne $keyHash) {
        throw 'DPAPI file changed during execution.'
    }
} finally {
    if ($bstr -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    $secureKey.Dispose()
    if ($process) { $process.Dispose() }
}
