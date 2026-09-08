@echo off
REM PhotoAI Backend - hizli/otomatik kurulum tetikleyicisi.
REM Asil mantik setup.ps1'de (PowerShell) - burasi sadece cift-tiklanabilir
REM bir giris noktasi, ExecutionPolicy kisitlamasina takilmadan calisir.
REM
REM Parametre gecmek icin (ornek CPU-only + Qdrant atla):
REM   setup.bat -Cpu -SkipQdrant
REM Secenekler icin: powershell -File setup.ps1 -? (ya da setup.ps1 basindaki
REM yorum blogu / KURULUM-HIZLI.md)

setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
echo.
pause
