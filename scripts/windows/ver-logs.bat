@echo off
title Verificador de Cartas - Logs
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0ps\logs.ps1"
echo.
pause
