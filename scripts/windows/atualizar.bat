@echo off
title Verificador de Cartas - Atualizar
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0ps\update.ps1"
echo.
pause
