[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $RepoRoot,
    [Parameter(Mandatory = $true)][string] $Manifest,
    [Parameter(Mandatory = $true)][string] $ReviewCsv,
    [Parameter(Mandatory = $true)][string] $BatchRoot,
    [Parameter(Mandatory = $true)][string] $PrivateRoot,
    [Parameter(Mandatory = $true)][string] $ExecutionRoot,
    [Parameter(Mandatory = $true)][string] $Authorization,
    [switch] $Live
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $Live) { throw 'This launcher only runs with the explicit -Live switch.' }

$repo = [IO.Path]::GetFullPath($RepoRoot)
$private = [IO.Path]::GetFullPath($PrivateRoot)
$batch = [IO.Path]::GetFullPath($BatchRoot)
$execution = [IO.Path]::GetFullPath($ExecutionRoot)
$manifestPath = [IO.Path]::GetFullPath($Manifest)
$reviewPath = [IO.Path]::GetFullPath($ReviewCsv)
$authorizationPath = [IO.Path]::GetFullPath($Authorization)
foreach ($path in @($batch, $execution, $manifestPath, $reviewPath, $authorizationPath)) {
    if (-not $path.StartsWith($private + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Batch inputs, authorization, and execution output must remain under the private asset root.'
    }
}
if (-not (Test-Path -LiteralPath $repo -PathType Container) -or
    -not (Test-Path -LiteralPath $manifestPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $reviewPath -PathType Leaf) -or
    -not (Test-Path -LiteralPath $authorizationPath -PathType Leaf)) {
    throw 'Required G3 live input is missing.'
}
if (Test-Path -LiteralPath $execution) { throw 'Execution root already exists; preserve it and choose a new attempt directory.' }

$auth = Get-Content -LiteralPath $authorizationPath -Raw | ConvertFrom-Json
$ids = @($auth.approved_source_ids | ForEach-Object { [string]$_ })
$expected = @('556', '640', '645')
if ($auth.record_type -cne 'G3C2_V2_BATCH_AUTHORIZATION' -or
    $auth.status -cne 'G3C2_V2_BATCH_AUTHORIZED' -or
    $auth.authorization_basis -cne 'EXPLICIT_USER_AUTHORIZATION_IN_CURRENT_TURN' -or
    $auth.gate_status -cne 'G3C2_FINAL_PREFLIGHT_PASS / BOUNDED' -or
    $auth.max_posts -ne 3 -or $auth.request_count_cap -ne 3 -or
    $ids.Count -ne 3 -or @($ids | Sort-Object) -join ',' -cne @($expected | Sort-Object) -join ',') {
    throw 'G3-C2 authorization must bind exactly the three approved sources and a three-POST maximum.'
}
$rows = @(Import-Csv -LiteralPath $manifestPath)
$manifestIds = @($rows | ForEach-Object { [string]$_.source_id })
if ($manifestIds.Count -ne 3 -or @($manifestIds | Sort-Object) -join ',' -cne @($expected | Sort-Object) -join ',') {
    throw 'G3-C2 request Manifest is not the exact authorized three-source set.'
}
$baseUrl = [Environment]::GetEnvironmentVariable('BORESCOPE_IMAGE_BASE_URL', 'User')
$model = [Environment]::GetEnvironmentVariable('BORESCOPE_IMAGE_MODEL', 'User')
if ($baseUrl -cne 'https://direct-api.ccccapi.cc' -or $model -cne 'gpt-image-2.5-sunburst') {
    throw 'Windows User relay settings do not match the frozen G2 protocol.'
}

$keyFile = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'BorescopeImageRelay\relay_key.dpapi'
if (-not (Test-Path -LiteralPath $keyFile -PathType Leaf)) { throw 'CurrentUser DPAPI key file unavailable.' }
$cipherHashBefore = (Get-FileHash -LiteralPath $keyFile -Algorithm SHA256).Hash
$secureKey = $null
$plainKey = $null
$bstr = [IntPtr]::Zero
$child = $null
$stdout = $null
$stderr = $null
try {
    $cipherLine = Get-Content -LiteralPath $keyFile -TotalCount 1
    if ([string]::IsNullOrWhiteSpace($cipherLine)) { throw 'CurrentUser DPAPI file is empty.' }
    $secureKey = ConvertTo-SecureString -String $cipherLine -ErrorAction Stop
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    if ([string]::IsNullOrWhiteSpace($plainKey)) { throw 'CurrentUser DPAPI decryption returned empty value.' }

    $python = (Get-Command py.exe -ErrorAction Stop).Source
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $python
    $startInfo.WorkingDirectory = $repo
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($arg in @('-3', '-m', 'tools.exp13_cloud.g3.g3_batch_executor',
        '--repo-root', $repo, '--manifest', $manifestPath, '--review-csv', $reviewPath,
        '--batch-root', $batch, '--private-root', $private, '--execution-root', $execution,
        '--authorization', $authorizationPath, '--live')) {
        $null = $startInfo.ArgumentList.Add($arg)
    }
    $startInfo.EnvironmentVariables['G3_IMAGE_API_KEY'] = $plainKey
    $child = [Diagnostics.Process]::new()
    $child.StartInfo = $startInfo
    if (-not $child.Start()) { throw 'G3 executor child failed to start.' }
    [void] $startInfo.EnvironmentVariables.Remove('G3_IMAGE_API_KEY')
    $plainKey = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    $bstr = [IntPtr]::Zero
    $secureKey.Dispose()
    $secureKey = $null

    $stdoutTask = $child.StandardOutput.ReadToEndAsync()
    $stderrTask = $child.StandardError.ReadToEndAsync()
    $child.WaitForExit()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    if ($child.ExitCode -ne 0) { throw 'G3 executor stopped; inspect private request evidence and never retry uncertain sources.' }
    $cipherHashAfter = (Get-FileHash -LiteralPath $keyFile -Algorithm SHA256).Hash
    if ($cipherHashBefore -cne $cipherHashAfter) { throw 'DPAPI file changed during execution.' }
    Write-Output $stdout
}
finally {
    if ($plainKey) {
        $plainKey = $null
    }
    if ($bstr -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
    if ($secureKey) { $secureKey.Dispose() }
    if ($child -and -not $child.HasExited) { $child.WaitForExit() }
    if ($stderr) { Write-Error 'G3 executor wrote private error output; inspect the protected execution directory.' -ErrorAction Continue }
}
