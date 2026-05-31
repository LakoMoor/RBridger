@echo off
setlocal

set WORKSPACE=%~dp0..\..
cd /d "%WORKSPACE%"

echo Building release binary...
cargo build --release -p rusty-bridge-ui
if errorlevel 1 goto :error

mkdir dist\out 2>nul

echo Running NSIS...
cd dist\windows
makensis installer.nsi
if errorlevel 1 goto :error

echo Done. Installer: dist\out\RustyBridge-0.1.0-windows-setup.exe
goto :eof

:error
echo Build failed.
exit /b 1
