[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $AssetRoot,
    [switch] $DryRun,
    [switch] $Live,
    [string] $ConfirmOnePaidPost = ''
)

& (Join-Path $PSScriptRoot 'Start-G2Source.ps1') -SourceId '23' -WrapperPath $PSCommandPath `
    -AssetRoot $AssetRoot -DryRun:$DryRun -Live:$Live -ConfirmOnePaidPost $ConfirmOnePaidPost
