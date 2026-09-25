param(
    [ValidateSet("web", "chat", "index", "evaluate", "test", "check")]
    [string]$Mode = "web",
    [int]$Port = 8000,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)
$ErrorActionPreference = "Stop"
$RagPython = "D:\Anaconda3\envs\lightrag\python.exe"
if (-not (Test-Path -LiteralPath $RagPython)) { throw "Python environment not found: $RagPython" }
$env:PYTHONIOENCODING = "utf-8"
Push-Location -LiteralPath $PSScriptRoot
try {
    switch ($Mode) {
        "web" { & $RagPython -m uvicorn rag_app.api:app --host 127.0.0.1 --port $Port --workers 1 @ExtraArgs }
        "chat" { & $RagPython -m rag_app.chat @ExtraArgs }
        "index" { & $RagPython -m rag_app.runtime @ExtraArgs }
        "evaluate" { & $RagPython -m rag_app.evaluate @ExtraArgs }
        "test" { & $RagPython -m unittest discover -s tests -p "test_*.py" @ExtraArgs }
        "check" { & $RagPython -m rag_app.check @ExtraArgs }
    }
    $RagExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $RagExitCode
