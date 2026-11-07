@echo off
title Verificador de Cartas - Parar
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0ps\stop.ps1"
echo.
pause
