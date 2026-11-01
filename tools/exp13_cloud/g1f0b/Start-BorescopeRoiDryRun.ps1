[CmdletBinding()]
param(
    [switch] $Live,
    [string] $PythonExecutable = 'C:\Users\kkkk\AppData\Local\Programs\Python\Python314\python.exe'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
# No live implementation exists in G1F-0B. Reject before touching credentials.
if ($Live) { throw 'G1F-0B is offline-only. Live requires separate user approval and a separate executor.' }

$secureKey = $null
$cipherLine = $null
$plainKey = $null
$bstr = [IntPtr]::Zero
$startInfo = $null
$child = $null
$result = $null
$stage = 'CONFIG_CHECK'
$success = $false
$runId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '_' + [Guid]::NewGuid().ToString('N')
$runDirectory = Join-Path (Join-Path $PSScriptRoot 'dry_runs') $runId
$resultPath = Join-Path $PSScriptRoot 'dpapi_loader_result.json'
if (Test-Path -LiteralPath $resultPath) {
    $resultPath = Join-Path $PSScriptRoot ('dpapi_loader_result_' + $runId + '.json')
}

try {
    if ($env:OS -ne 'Windows_NT') { throw 'Windows required' }
    $baseUrl = [Environment]::GetEnvironmentVariable('BORESCOPE_IMAGE_BASE_URL', 'User')
    $model = [Environment]::GetEnvironmentVariable('BORESCOPE_IMAGE_MODEL', 'User')
    if ($baseUrl -cne 'https://direct-api.ccccapi.cc' -or $model -cne 'gpt-image-2.5-sunburst') {
        throw 'Unexpected user-level relay configuration'
    }
    $childScript = Join-Path $PSScriptRoot 'verify_dpapi_dry_run.py'
    if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) { throw 'Python executable unavailable' }
    if (-not (Test-Path -LiteralPath $childScript -PathType Leaf)) { throw 'Fixed offline script unavailable' }
    $keyFile = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'BorescopeImageRelay\relay_key.dpapi'
    if (-not (Test-Path -LiteralPath $keyFile -PathType Leaf)) { throw 'DPAPI file unavailable' }
    $cipherHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $keyFile).Hash
    $stage = 'DPAPI_DECRYPT'
    # No -Key / -SecureKey: Windows CurrentUser DPAPI decryption.
    $cipherLine = Get-Content -LiteralPath $keyFile -TotalCount 1 -ErrorAction Stop
    if ([string]::IsNullOrWhiteSpace($cipherLine)) { throw 'Empty DPAPI input' }
    $secureKey = ConvertTo-SecureString -String $cipherLine -ErrorAction Stop
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    if ([string]::IsNullOrWhiteSpace($plainKey)) { throw 'Empty decrypted key' }
    $stage = 'OFFLINE_CHILD'
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $PythonExecutable
    $startInfo.Arguments = '-I "' + $childScript + '" --dry-run --output-directory "' + $runDirectory + '"'
    $startInfo.WorkingDirectory = $PSScriptRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    # Only the fixed Python child's environment receives the plaintext key.
    $startInfo.EnvironmentVariables['OPENAI_API_KEY'] = $plainKey
    $startInfo.EnvironmentVariables['OPENAI_BASE_URL'] = $baseUrl
    $startInfo.EnvironmentVariables['OPENAI_IMAGE_MODEL'] = $model
    $child = [Diagnostics.Process]::new()
    $child.StartInfo = $startInfo
    if (-not $child.Start()) { throw 'Child start failed' }
    [void] $startInfo.EnvironmentVariables.Remove('OPENAI_API_KEY')
    $plainKey = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    $bstr = [IntPtr]::Zero
    $secureKey.Dispose()
    $secureKey = $null
    $cipherLine = $null
    # Consume but never echo child output; report only allowlisted safe fields.
    $stdoutTask = $child.StandardOutput.ReadToEndAsync()
    $stderrTask = $child.StandardError.ReadToEndAsync()
    $child.WaitForExit()
    [void] $stdoutTask.GetAwaiter().GetResult()
    [void] $stderrTask.GetAwaiter().GetResult()
    if ($child.ExitCode -ne 0) { throw 'Offline child failed' }
    $reportPath = Join-Path $runDirectory 'dry_run_report.json'
    $childReport = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
    if ($childReport.status -ne 'DRY_RUN_PASS' -or $childReport.api_post_count -ne 0) {
        throw 'Invalid dry run result'
    }
    $cipherHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $keyFile).Hash
    if ($cipherHashBefore -ne $cipherHashAfter) { throw 'DPAPI file changed' }
    $result = [ordered]@{
        status = 'DPAPI_LOADER_DRY_RUN_PASS'
        base_url = $baseUrl
        model = $model
        configuration_source = 'Windows User environment'
        dpapi_scope = 'CurrentUser'
        dpapi_first_line_only = $true
        dpapi_decryption_succeeded = $true
        dpapi_file_unchanged = $true
        credential_present_in_python_child = [bool] $childReport.credential_present_in_child
        credential_removed_from_child_environment = [bool] $childReport.credential_removed_from_child_environment
        parent_environment_mutated = $false
        persistent_openai_key_written = $false
        key_in_command_line = $false
        key_logged = $false
        child_report_path = $reportPath
        child_exit_code = $child.ExitCode
        api_post_count = 0
        network_requests = 0
        live_supported = $false
    }
    $success = $true
} catch {
    # Never emit raw exception records from a credential-bearing operation.
    $result = [ordered]@{
        status = 'DPAPI_LOADER_DRY_RUN_FAILED'
        failed_stage = $stage
        api_post_count = 0
        network_requests = 0
        key_logged = $false
    }
} finally {
    if ($null -ne $startInfo) { [void] $startInfo.EnvironmentVariables.Remove('OPENAI_API_KEY') }
    $plainKey = $null
    if ($bstr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        $bstr = [IntPtr]::Zero
    }
    if ($null -ne $secureKey) { $secureKey.Dispose(); $secureKey = $null }
    $cipherLine = $null
    if ($null -ne $child) { $child.Dispose(); $child = $null }
    $startInfo = $null
    $stdoutTask = $null
    $stderrTask = $null
    $childReport = $null
}
$result['temporary_sensitive_references_cleared'] = $true
$jsonText = $result | ConvertTo-Json -Depth 6
$resultBytes = [Text.UTF8Encoding]::new($false).GetBytes($jsonText + [Environment]::NewLine)
$fileStream = [IO.File]::Open($resultPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try { $fileStream.Write($resultBytes, 0, $resultBytes.Length) } finally { $fileStream.Dispose() }
if (-not $success) { throw ('Offline DPAPI validation failed at ' + $stage + '; no request sent; see sanitized result.') }
Write-Output ('DPAPI_LOADER_DRY_RUN_PASS; API_POST=0; report=' + $resultPath)
