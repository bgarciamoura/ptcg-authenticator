@echo off
title Verificador de Cartas - Iniciar
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0ps\start.ps1"
echo.
pause
