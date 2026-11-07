@echo off
title Verificador de Cartas - Status
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0ps\status.ps1"
echo.
pause
