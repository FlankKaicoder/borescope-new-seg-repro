[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Python,
    [ValidateRange(1,8)][int] $Workers = 1
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$base = [Environment]::GetEnvironmentVariable('BORESCOPE_IMAGE_BASE_URL', 'User')
$model = [Environment]::GetEnvironmentVariable('BORESCOPE_IMAGE_MODEL', 'User')
if ($base -cne 'https://direct-api.ccccapi.cc' -or $model -cne 'gpt-image-2.5-sunburst') {
    throw 'Windows User relay settings differ from the frozen G2 protocol.'
}
$keyFile = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'BorescopeImageRelay\relay_key.dpapi'
$keyHash = (Get-FileHash -LiteralPath $keyFile -Algorithm SHA256).Hash
$secureKey = $null
$bstr = [IntPtr]::Zero
$process = $null
try {
    $cipher = Get-Content -LiteralPath $keyFile -TotalCount 1
    $secureKey = ConvertTo-SecureString -String $cipher
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = (Resolve-Path -LiteralPath $Python).Path
    $startInfo.WorkingDirectory = $repo
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    foreach ($argument in @('-u', '-m', 'tools.exp14.run_large_scale_repair', '--live', '--workers', "$Workers")) {
        $startInfo.ArgumentList.Add($argument)
    }
    $startInfo.Environment['EXP14_IMAGE_API_KEY'] = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    try {
        $process = [Diagnostics.Process]::Start($startInfo)
    } finally {
        [void] $startInfo.Environment.Remove('EXP14_IMAGE_API_KEY')
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        $bstr = [IntPtr]::Zero
        $secureKey.Dispose()
        $secureKey = $null
    }
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw 'Exp14 runner stopped. Preserve all locks and inspect private logs.' }
    if ((Get-FileHash -LiteralPath $keyFile -Algorithm SHA256).Hash -cne $keyHash) {
        throw 'DPAPI file changed during execution.'
    }
} finally {
    if ($bstr -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    if ($secureKey) { $secureKey.Dispose() }
    if ($process) { $process.Dispose() }
}
