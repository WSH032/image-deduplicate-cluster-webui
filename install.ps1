function InstallFail {
    Write-Output "Install failed."
    Read-Host | Out-Null ;
    Exit
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}

if (!(Test-Path -Path "venv")) {
    Write-Output "Creating venv..."
    python -m venv venv
    Check "Creating venv failed, please check your python."
}

.\venv\Scripts\activate
Check "Activate venv failed."

Write-Output "pip installing..."
pip install -r requirements.txt
Check "pip install failed."

Write-Output "All done"
Read-Host | Out-Null ;
