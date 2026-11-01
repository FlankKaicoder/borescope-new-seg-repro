[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet('260', '470', '23')][string] $SourceId,
    [Parameter(Mandatory = $true)][string] $WrapperPath,
    [Parameter(Mandatory = $true)][string] $AssetRoot,
    [switch] $DryRun,
    [switch] $Live,
    [string] $ConfirmOnePaidPost = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($DryRun -and $Live) { throw 'Choose DryRun or Live, not both.' }
if (-not $DryRun -and -not $Live) { $DryRun = $true }

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$bundlePath = Join-Path $repoRoot 'configs\exp13_cloud\g2\execution_bundle.json'
$pythonScript = Join-Path $PSScriptRoot 'g2_single_executor.py'
$protocolPath = Join-Path $repoRoot 'configs\exp13_cloud\g2\pilot_protocol_v2.json'
$assetRoot = [IO.Path]::GetFullPath($AssetRoot)
$executionRoot = Join-Path (Split-Path -Parent $assetRoot) 'cloud_pilot_g2_execution_20261102'
$liveSourceRoot = Join-Path (Join-Path $executionRoot 'live') $SourceId
$sourceLockRoot = Join-Path (Join-Path $executionRoot 'locks') $SourceId
$startedPath = Join-Path $sourceLockRoot 'STARTED.json'
$confirmations = @{
    '260' = 'G2_CRACK260_ONE_PAID_POST'
    '470' = 'G2_CRACK470_ONE_PAID_POST'
    '23' = 'G2_CORROSION23_ONE_PAID_POST'
}
$roles = @{
    '260' = 'NEW_CLOUD_SOURCE'
    '470' = 'NEW_CLOUD_SOURCE'
    '23' = 'HISTORICAL_FAILURE_TARGETED_RETEST'
}
$stage = 'STATIC_CONFIGURATION_CHECK'
$success = $false
$secureKey = $null
$cipherLine = $null
$plainKey = $null
$bstr = [IntPtr]::Zero
$startInfo = $null
$child = $null
$childOut = $null
$childErr = $null
$stdoutTask = $null
$stderrTask = $null
$childReport = $null
$keyFile = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'BorescopeImageRelay\relay_key.dpapi'

try {
    if (-not (Test-Path -LiteralPath $bundlePath -PathType Leaf)) { throw 'Frozen execution bundle missing.' }
    $bundleBytes = [IO.File]::ReadAllBytes($bundlePath)
    $bundleSha = (Get-FileHash -LiteralPath $bundlePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $bundle = [Text.Encoding]::UTF8.GetString($bundleBytes) | ConvertFrom-Json
    $wrapperRelative = 'tools/exp13_cloud/g2/' + [IO.Path]::GetFileName($WrapperPath)
    $wrapperEntry = $bundle.files | Where-Object { $_.path -ceq $wrapperRelative } | Select-Object -First 1
    if (-not $wrapperEntry) { throw 'Fixed source wrapper is absent from execution bundle.' }
    $wrapperSha = (Get-FileHash -LiteralPath $WrapperPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($wrapperSha -cne $wrapperEntry.sha256) { throw 'Source wrapper SHA256 mismatch.' }
    foreach ($entry in $bundle.files) {
        $filePath = Join-Path $repoRoot ($entry.path -replace '/', '\')
        if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) { throw 'Frozen bundle member missing.' }
        if ((Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $entry.sha256) {
            throw 'Frozen bundle member SHA256 mismatch.'
        }
    }
    if ($bundle.model -cne 'gpt-image-2.5-sunburst' -or $bundle.base_url -cne 'https://direct-api.ccccapi.cc') {
        throw 'Frozen relay configuration mismatch.'
    }
    $sourceEntry = $bundle.sources.PSObject.Properties[$SourceId].Value
    if (-not $sourceEntry -or $sourceEntry.source_role -cne $roles[$SourceId]) { throw 'Source role mismatch.' }
    if (-not (Test-Path -LiteralPath (Join-Path $assetRoot 'private_three_source_manifest.json') -PathType Leaf)) {
        throw 'Frozen private input manifest unavailable.'
    }
    if (Test-Path -LiteralPath $startedPath -PathType Leaf) { throw 'Permanent STARTED marker exists; never resend.' }
    if (Test-Path -LiteralPath $sourceLockRoot) { throw 'Permanent source lock path exists; never reuse.' }
    if (Test-Path -LiteralPath $liveSourceRoot) { throw 'Source live output path already exists; never reuse.' }
    $startedCount = 0
    foreach ($id in @('260', '470', '23')) {
        if (Test-Path -LiteralPath (Join-Path (Join-Path $executionRoot 'locks') $id)) { $startedCount++ }
    }
    if ($startedCount -ge 3) { throw 'G2 POST budget is occupied.' }

    if ($Live) {
        if ($ConfirmOnePaidPost -cne $confirmations[$SourceId]) { throw 'Source-specific paid-call confirmation missing.' }
        $receiptPath = Join-Path (Join-Path (Join-Path $executionRoot 'authorizations') $SourceId) 'gate_approval.json'
        if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) { throw 'Source-specific Gate receipt missing.' }
        $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
        if ($receipt.status -cne 'G2_SOURCE_LIVE_APPROVED' -or -not $receipt.independent_gate_pass -or -not $receipt.user_approved) {
            throw 'Independent ChatGPT Gate/user approval missing.'
        }
        if ($receipt.source_id -cne $SourceId -or $receipt.source_role -cne $roles[$SourceId] -or $receipt.model -cne 'gpt-image-2.5-sunburst' -or $receipt.max_posts -ne 1 -or $receipt.execution_bundle_sha256 -cne $bundleSha) { throw 'Gate receipt scope/bundle mismatch.' }
        if ([string]::IsNullOrWhiteSpace($receipt.approval_reference)) { throw 'Approval reference missing.' }
    }

    $baseUrl = [Environment]::GetEnvironmentVariable('BORESCOPE_IMAGE_BASE_URL', 'User')
    $model = [Environment]::GetEnvironmentVariable('BORESCOPE_IMAGE_MODEL', 'User')
    if ($baseUrl -cne 'https://direct-api.ccccapi.cc' -or $model -cne 'gpt-image-2.5-sunburst') {
        throw 'Windows User relay settings do not match the frozen G2 protocol.'
    }
    $python = (Get-Command py.exe -ErrorAction Stop).Source
    if (-not (Test-Path -LiteralPath $pythonScript -PathType Leaf)) { throw 'G2 executor unavailable.' }
    if (-not (Test-Path -LiteralPath $keyFile -PathType Leaf)) { throw 'CurrentUser DPAPI key file unavailable.' }
    $cipherHashBefore = (Get-FileHash -LiteralPath $keyFile -Algorithm SHA256).Hash

    $runId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '_' + [Guid]::NewGuid().ToString('N')
    $dryRunDir = Join-Path (Join-Path (Join-Path $executionRoot 'dry_run') $SourceId) $runId
    $loaderReportPath = Join-Path $dryRunDir 'dpapi_loader_report.json'

    $stage = 'DPAPI_CURRENT_USER_DECRYPT'
    $cipherLine = Get-Content -LiteralPath $keyFile -TotalCount 1 -ErrorAction Stop
    if ([string]::IsNullOrWhiteSpace($cipherLine)) { throw 'DPAPI file first line is empty.' }
    $secureKey = ConvertTo-SecureString -String $cipherLine -ErrorAction Stop
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    if ([string]::IsNullOrWhiteSpace($plainKey)) { throw 'CurrentUser DPAPI decryption returned empty value.' }

    $stage = 'FROZEN_CHILD_DRY_RUN'
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $python
    $startInfo.WorkingDirectory = $repoRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $null = $startInfo.ArgumentList.Add('-3')
    $null = $startInfo.ArgumentList.Add('-I')
    $null = $startInfo.ArgumentList.Add($pythonScript)
    $null = $startInfo.ArgumentList.Add($(if ($Live) {'--live'} else {'--dry-run'}))
    $null = $startInfo.ArgumentList.Add('--source-id')
    $null = $startInfo.ArgumentList.Add($SourceId)
    $null = $startInfo.ArgumentList.Add('--asset-root')
    $null = $startInfo.ArgumentList.Add($assetRoot)
    $null = $startInfo.ArgumentList.Add('--execution-root')
    $null = $startInfo.ArgumentList.Add($executionRoot)
    if ($Live) {
        $null = $startInfo.ArgumentList.Add('--confirm-one-paid-post')
        $null = $startInfo.ArgumentList.Add($ConfirmOnePaidPost)
    } else {
        $null = $startInfo.ArgumentList.Add('--output-dir')
        $null = $startInfo.ArgumentList.Add($dryRunDir)
    }
    $startInfo.EnvironmentVariables['OPENAI_API_KEY'] = $plainKey
    $startInfo.EnvironmentVariables['OPENAI_BASE_URL'] = $baseUrl
    $startInfo.EnvironmentVariables['OPENAI_IMAGE_MODEL'] = $model
    $child = [Diagnostics.Process]::new()
    $child.StartInfo = $startInfo
    if (-not $child.Start()) { throw 'Frozen Python child failed to start.' }
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
    $childOut = $stdoutTask.GetAwaiter().GetResult()
    $childErr = $stderrTask.GetAwaiter().GetResult()
    if ($child.ExitCode -ne 0) { throw 'G2 Python child stopped; do not retry a live call.' }
    if (-not $Live) {
        $childReport = Get-Content -LiteralPath (Join-Path $dryRunDir 'dry_run_manifest.json') -Raw | ConvertFrom-Json
        if ($childReport.status -cne 'DRY_RUN_PASS / LIVE_NOT_STARTED' -or $childReport.api_post_count -ne 0 -or -not $childReport.credential_present_in_child) { throw 'G2 dry-run verification failed.' }
    }
    $cipherHashAfter = (Get-FileHash -LiteralPath $keyFile -Algorithm SHA256).Hash
    if ($cipherHashBefore -cne $cipherHashAfter) { throw 'DPAPI file changed during loader check.' }
    $report = [ordered]@{
        status = $(if ($Live) {'G2_SOURCE_SINGLE_LIVE_TERMINAL'} else {'G2_SOURCE_DPAPI_DRY_RUN_PASS'})
        source_id = $SourceId
        source_role = $roles[$SourceId]
        live_mode = [bool]$Live
        base_url = $baseUrl
        model = $model
        configuration_source = 'Windows User environment'
        dpapi_scope = 'CurrentUser'
        dpapi_first_line_only = $true
        dpapi_decryption_succeeded = $true
        dpapi_file_unchanged = $true
        credential_present_in_python_child = $(if ($Live) {$true} else {[bool]$childReport.credential_present_in_child})
        credential_removed_from_child_environment = $(if ($Live) {$true} else {[bool]$childReport.credential_removed_from_child_environment})
        parent_environment_mutated = $false
        persistent_openai_key_written = $false
        key_in_command_line_or_logs = $false
        child_exit_code = $child.ExitCode
        child_report_path = $(if ($Live) {Join-Path $liveSourceRoot 'TERMINAL.json'} else {Join-Path $dryRunDir 'dry_run_manifest.json'})
        api_post_count = $(if ($Live) {1} else {0})
        automatic_retry = 0
    }
    if (-not $Live) {
        $reportBytes = [Text.UTF8Encoding]::new($false).GetBytes(($report | ConvertTo-Json -Depth 6) + [Environment]::NewLine)
        [IO.Directory]::CreateDirectory($dryRunDir) | Out-Null
        $stream = [IO.File]::Open($loaderReportPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
        try { $stream.Write($reportBytes, 0, $reportBytes.Length) } finally { $stream.Dispose() }
    }
    $success = $true
} catch {
    $safeReport = [ordered]@{
        status = 'G2_PREFLIGHT_STOP_NO_AUTOMATIC_RETRY'
        source_id = $SourceId
        stage = $stage
        live_mode = [bool]$Live
        api_post_count = 0
        key_logged = $false
    }
    if ($DryRun -or -not $Live) {
        try {
            [IO.Directory]::CreateDirectory($dryRunDir) | Out-Null
            $failurePath = Join-Path $dryRunDir 'loader_failure_sanitized.json'
            $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($safeReport | ConvertTo-Json -Depth 5) + [Environment]::NewLine)
            $fs = [IO.File]::Open($failurePath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
            try { $fs.Write($bytes, 0, $bytes.Length) } finally { $fs.Dispose() }
        } catch { }
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
    $childOut = $null
    $childErr = $null
    $childReport = $null
}
if (-not $success) { throw 'G2 source preflight stopped; no automatic retry; inspect only sanitized local records.' }
if ($Live) {
    Write-Output ('G2_SOURCE_LIVE_TERMINAL source=' + $SourceId + '; API_POST=1; STOP')
} else {
    Write-Output ('G2_SOURCE_DPAPI_DRY_RUN_PASS source=' + $SourceId + '; API_POST=0; report=' + $loaderReportPath)
}
