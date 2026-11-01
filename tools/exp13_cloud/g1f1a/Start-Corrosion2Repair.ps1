[CmdletBinding()]
param(
    [switch] $Live,
    [string] $ConfirmSinglePaidRequest = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$receiptPath = Join-Path $PSScriptRoot 'gate_approval.json'
if ($Live) {
    # Independent approval is deliberately absent during G1F-1A.
    if ($ConfirmSinglePaidRequest -cne 'AUTHORIZE_G1F1_CORROSION2_ONE_PAID_POST') {
        throw 'Explicit one-paid-request confirmation required; no credential loaded.'
    }
    if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
        throw 'Independent Gate receipt missing; LIVE_HOLD; no credential loaded.'
    }
    $receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
    if ($receipt.status -cne 'G1F_1_LIVE_APPROVED' -or -not $receipt.independent_gate_pass -or -not $receipt.user_approved) {
        throw 'Independent Gate/user approval missing; LIVE_HOLD; no credential loaded.'
    }
    if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'STARTED.json')) {
        throw 'Permanent request budget already occupied; never resend.'
    }
}
$secureKey = $null
$cipherLine = $null
$plainKey = $null
$bstr = [IntPtr]::Zero
$startInfo = $null
$child = $null
$stdoutTask = $null
$stderrTask = $null
$stage = 'CONFIG_CHECK'
$success = $false
$runId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '_' + [Guid]::NewGuid().ToString('N')
$runDirectory = Join-Path (Join-Path $PSScriptRoot 'dry_runs') $runId
$resultPath = Join-Path $PSScriptRoot ('loader_result_' + $runId + '.json')
try {
    $baseUrl = [Environment]::GetEnvironmentVariable('BORESCOPE_IMAGE_BASE_URL', 'User')
    $model = [Environment]::GetEnvironmentVariable('BORESCOPE_IMAGE_MODEL', 'User')
    if ($baseUrl -cne 'https://direct-api.ccccapi.cc' -or $model -cne 'gpt-image-2.5-sunburst') {
        throw 'Unexpected persistent relay configuration'
    }
    $python = 'C:\Users\kkkk\AppData\Local\Programs\Python\Python314\python.exe'
    $script = Join-Path $PSScriptRoot 'single_repair.py'
    $keyFile = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'BorescopeImageRelay\relay_key.dpapi'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Python unavailable' }
    $cipherHashBefore = (Get-FileHash -Algorithm SHA256 -LiteralPath $keyFile).Hash
    $stage = 'DPAPI_DECRYPT'
    # Reused G1F-0B CurrentUser DPAPI procedure; existing file is never rewritten.
    $cipherLine = Get-Content -LiteralPath $keyFile -TotalCount 1 -ErrorAction Stop
    if ([string]::IsNullOrWhiteSpace($cipherLine)) { throw 'Empty DPAPI input' }
    $secureKey = ConvertTo-SecureString -String $cipherLine -ErrorAction Stop
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    if ([string]::IsNullOrWhiteSpace($plainKey)) { throw 'Empty key' }
    $stage = 'PYTHON_CHILD'
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $python
    $startInfo.Arguments = '-I "' + $script + '"'
    if ($Live) {
        $startInfo.Arguments += ' --live --confirm AUTHORIZE_G1F1_CORROSION2_ONE_PAID_POST --gate-receipt "' + $receiptPath + '"'
    } else {
        $startInfo.Arguments += ' --dry-run --output-directory "' + $runDirectory + '"'
    }
    $startInfo.WorkingDirectory = $PSScriptRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.EnvironmentVariables['OPENAI_API_KEY'] = $plainKey
    $startInfo.EnvironmentVariables['OPENAI_BASE_URL'] = $baseUrl
    $startInfo.EnvironmentVariables['OPENAI_IMAGE_MODEL'] = $model
    $child = [Diagnostics.Process]::new()
    $child.StartInfo = $startInfo
    if (-not $child.Start()) { throw 'Child failed to start' }
    [void] $startInfo.EnvironmentVariables.Remove('OPENAI_API_KEY')
    $plainKey = $null
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    $bstr = [IntPtr]::Zero
    $secureKey.Dispose()
    $secureKey = $null
    $cipherLine = $null
    $stdoutTask = $child.StandardOutput.ReadToEndAsync()
    $stderrTask = $child.StandardError.ReadToEndAsync()
    $child.WaitForExit()
    [void] $stdoutTask.GetAwaiter().GetResult()
    [void] $stderrTask.GetAwaiter().GetResult()
    if ($child.ExitCode -ne 0) { throw 'Child stopped; never automatically retry live' }
    $cipherHashAfter = (Get-FileHash -Algorithm SHA256 -LiteralPath $keyFile).Hash
    if ($cipherHashBefore -ne $cipherHashAfter) { throw 'DPAPI file changed' }
    $childReportPath = if ($Live) { Join-Path $PSScriptRoot 'live_result\TERMINAL.json' } else { Join-Path $runDirectory 'dry_run_report.json' }
    $childReport = Get-Content -Raw -LiteralPath $childReportPath | ConvertFrom-Json
    if (-not $Live -and $childReport.status -cne 'DRY_RUN_PASS') { throw 'Dry run did not pass' }
    $result = [ordered]@{
        status = $(if ($Live) {'SINGLE_LIVE_TERMINAL'} else {'DPAPI_LOADER_DRY_RUN_PASS'})
        live_mode = [bool] $Live
        base_url = $baseUrl
        model = $model
        configuration_source = 'Windows User'
        dpapi_current_user_decryption = $true
        dpapi_first_line_only = $true
        dpapi_file_unchanged = $true
        credential_present_in_child = $(if ($Live) {$true} else {[bool]$childReport.credential_present_in_child})
        child_report_path = $childReportPath
        parent_environment_mutated = $false
        persistent_openai_key_written = $false
        key_in_command_line_or_logs = $false
        post_budget_occupied = [bool](Test-Path -LiteralPath (Join-Path $PSScriptRoot 'STARTED.json'))
        api_post_count = $(if ($Live) {1} else {0})
        retry = 0
    }
    $success = $true
} catch {
    $result = [ordered]@{
        status = 'STOP_FAILED_NO_AUTOMATIC_RETRY'
        stage = $stage
        live_mode = [bool] $Live
        post_budget_occupied = [bool](Test-Path -LiteralPath (Join-Path $PSScriptRoot 'STARTED.json'))
        api_post_count = $(if ($Live) {'UNKNOWN_CHECK_MARKER_AND_LOCAL_LEDGER'} else {0})
        key_logged = $false
    }
} finally {
    if ($null -ne $startInfo) { [void] $startInfo.EnvironmentVariables.Remove('OPENAI_API_KEY') }
    $plainKey = $null
    if ($bstr -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr); $bstr = [IntPtr]::Zero }
    if ($null -ne $secureKey) { $secureKey.Dispose(); $secureKey = $null }
    $cipherLine = $null
    if ($null -ne $child) { $child.Dispose(); $child = $null }
    $startInfo = $null
    $stdoutTask = $null
    $stderrTask = $null
    $childReport = $null
}
$result['temporary_sensitive_references_cleared'] = $true
$jsonText = $result | ConvertTo-Json -Depth 8
$bytes = [Text.UTF8Encoding]::new($false).GetBytes($jsonText + [Environment]::NewLine)
$stream = [IO.File]::Open($resultPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
try { $stream.Write($bytes, 0, $bytes.Length) } finally { $stream.Dispose() }
if (-not $success) { throw ('Stopped at ' + $stage + '; do not retry a live call; inspect local sanitized ledger.') }
Write-Output ('Completed ' + $result.status + '; report=' + $resultPath)
