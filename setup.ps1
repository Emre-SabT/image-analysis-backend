<#
.SYNOPSIS
    PhotoAI Backend - otomatik kurulum scripti (Windows / PowerShell).

.DESCRIPTION
    KURULUM-HIZLI.md'deki adimlarin otomatiklestirilmis hali. Idempotenttir:
    zaten yapilmis adimlari atlar, mevcut .env'in UZERINE YAZMAZ. Basarisiz
    olan opsiyonel adimlar (DB, Qdrant indirme) scripti durdurmaz - uyari
    basip devam eder, sonda ozet gosterir.

    PostgreSQL kurulu degilse EDB'nin resmi installer'ini (unattended mod)
    indirip kurar - bunun icin YONETICI ONAYI (UAC) ister; parola installer'a
    komut satiri argumani olarak gecilir (EDB'nin resmi --superpassword
    mekanizmasi budur), kurulum sirasinda process komut satirinda kisaca
    gorunur olabilir.

    AWS/Bedrock kimlik bilgilerine hic DOKUNMAZ (boto3 kendi zincirini
    kullanir - bkz. README.md). Bu script sadece LOKAL dosya/DB kurulumunu
    otomatize eder.

.PARAMETER Cpu
    GPU/CUDA yerine CPU-only PyTorch kurar ve .env'de EMBEDDING_DEVICE=cpu
    yazar. Belirtilmezse nvidia-smi varligina gore otomatik karar verilir.

.PARAMETER SkipModels
    Model dosyalarinin indirilmesini atlar (zaten elde mevcutsa).

.PARAMETER SkipQdrant
    Qdrant indirme/kurulumunu atlar (zaten C:\qdrant'ta kuruluysa
    ya da Docker/baska bir makinede calisiyorsa).

.PARAMETER SkipDb
    PostgreSQL veritabani olusturma adimini atlar (kurulumu da).

.PARAMETER SkipPgInstall
    PostgreSQL zaten kurulu degilse OTOMATIK KURMAYI atlar (sadece DB
    olusturmayi degil) - PostgreSQL'in elle kurulacagi durumlar icin.
    Yonetici (admin) onayi/UAC istemeden calismasi gerekenler bunu kullanmali.

.PARAMETER Lan
    .env olusturulurken CORS_ORIGINS icin LAN adresi de sorar
    (bkz. README.md "LAN Uzerinden Erisim (on-prem)").

.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -Cpu -SkipQdrant
#>

param(
    [switch]$Cpu,
    [switch]$SkipModels,
    [switch]$SkipQdrant,
    [switch]$SkipDb,
    [switch]$SkipPgInstall,
    [switch]$Lan
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

# PostgreSQL otomatik-kurulum surumu - periyodik olarak guncel bir surume
# bumplenmeli (bkz. https://www.enterprisedb.com/downloads/postgres-postgresql-downloads).
# Dogrulama: bu URL'in HTTP 200 dondugu 2026-09-08'de kontrol edildi.
$PgInstallerVersion = "17.7-1"
$PgMajorVersion = "17"
$PgInstallerUrl = "https://get.enterprisedb.com/postgresql/postgresql-$PgInstallerVersion-windows-x64.exe"

$warnings = New-Object System.Collections.Generic.List[string]

function Write-Step($n, $text) {
    Write-Host ""
    Write-Host "== [$n] $text ==" -ForegroundColor Cyan
}
function Write-Ok($text)   { Write-Host "  OK   $text" -ForegroundColor Green }
function Write-Skip($text) { Write-Host "  --   $text (atlandi)" -ForegroundColor DarkGray }
function Write-Warn2($text) {
    Write-Host "  UYARI $text" -ForegroundColor Yellow
    $warnings.Add($text) | Out-Null
}

# --- 0) Sanity: repo kokunde miyiz? ------------------------------------
Write-Step 0 "On kontrol"
if (-not (Test-Path (Join-Path $root "requirements.txt"))) {
    Write-Host "HATA: requirements.txt bulunamadi - bu script'i repo kokunden calistirin." -ForegroundColor Red
    exit 1
}
$pyVersion = (python --version) 2>&1
Write-Ok "Python: $pyVersion"
if ($pyVersion -notmatch "3\.11") {
    Write-Warn2 "Python 3.11 degil ($pyVersion) - torch==2.5.1+cu121 tekerlegi cp311 icindir, uyumsuzluk cikabilir."
}

# --- 1) Sanal ortam ------------------------------------------------------
Write-Step 1 "Python sanal ortami"
$venvActivate = Join-Path $root "venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Skip "venv\ zaten var"
} else {
    python -m venv venv
    Write-Ok "venv olusturuldu"
}
. $venvActivate
python -m pip install --upgrade pip --quiet
Write-Ok "venv aktif: $(python -c 'import sys; print(sys.executable)')"

# --- 2) Bagimliliklar -----------------------------------------------------
Write-Step 2 "Bagimliliklar (pip)"
$hasGpu = -not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
$torchOk = python -c "import torch" 2>$null; $torchInstalled = $LASTEXITCODE -eq 0

if ($torchInstalled) {
    Write-Skip "torch zaten kurulu"
} elseif ($hasGpu) {
    Write-Host "  GPU tespit edildi (nvidia-smi) - torch==2.5.1+cu121 kuruluyor..." -ForegroundColor DarkCyan
    pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "  GPU bulunamadi ya da -Cpu verildi - CPU-only torch kuruluyor..." -ForegroundColor DarkCyan
    (Get-Content requirements.txt) -replace 'torch==2\.5\.1\+cu121', 'torch==2.5.1' |
        Set-Content requirements-cpu.txt -Encoding utf8
    pip install -r requirements-cpu.txt
}
if (-not $torchInstalled) {
    python -c "import fastapi, sqlalchemy, cv2, onnxruntime, insightface, torch, qdrant_client; print('kutuphaneler OK')"
}
$cudaAvailable = (python -c "import torch; print(torch.cuda.is_available())") 2>$null
Write-Ok "torch hazir (cuda: $cudaAvailable)"

# --- 3) PostgreSQL (kurulum + veritabani) ----------------------------------
$dbPassword = $null
Write-Step 3 "PostgreSQL"
if ($SkipDb) {
    Write-Skip "-SkipDb verildi"
} else {
    function Find-Psql {
        $cmd = Get-Command psql -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
        $found = Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\psql.exe" -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | Select-Object -First 1
        if ($found) {
            $env:Path += ";$($found.DirectoryName)"
            return $found.FullName
        }
        return $null
    }

    $psqlPath = Find-Psql
    $freshInstall = $false

    if (-not $psqlPath -and -not $SkipPgInstall) {
        Write-Host "  PostgreSQL bulunamadi - otomatik kurulum baslatilacak (surum $PgInstallerVersion)." -ForegroundColor DarkCyan
        Write-Host "  Kurulum icin YONETICI ONAYI (UAC) istenecek." -ForegroundColor DarkCyan
        $secure = Read-Host "  Yeni 'postgres' super kullanicisi icin bir parola belirleyin" -AsSecureString
        $dbPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
            [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
        try {
            $installerPath = Join-Path $env:TEMP "postgresql-$PgInstallerVersion-windows-x64.exe"
            if (-not (Test-Path $installerPath)) {
                Write-Host "  Indiriliyor: $PgInstallerUrl" -ForegroundColor DarkCyan
                Invoke-WebRequest -Uri $PgInstallerUrl -OutFile $installerPath
            }
            $installArgs = @(
                "--mode", "unattended",
                "--unattendedmodeui", "minimal",
                "--superpassword", $dbPassword,
                "--serverport", "5432",
                "--servicename", "postgresql-x64-$PgMajorVersion"
            )
            Write-Host "  Kuruluyor (birkac dakika surebilir)..." -ForegroundColor DarkCyan
            $proc = Start-Process -FilePath $installerPath -ArgumentList $installArgs -Verb RunAs -Wait -PassThru
            if ($proc.ExitCode -ne 0) {
                throw "Installer exit code: $($proc.ExitCode)"
            }
            $env:Path += ";C:\Program Files\PostgreSQL\$PgMajorVersion\bin"
            $freshInstall = $true
            Write-Ok "PostgreSQL $PgMajorVersion kuruldu"

            # Servis ayaga kalkana kadar kisa bir bekleme/deneme dongusu.
            $ready = $false
            for ($i = 0; $i -lt 10 -and -not $ready; $i++) {
                Start-Sleep -Seconds 2
                $env:PGPASSWORD = $dbPassword
                & psql -U postgres -tAc "SELECT 1" 2>$null | Out-Null
                if ($LASTEXITCODE -eq 0) { $ready = $true }
            }
            if (-not $ready) { Write-Warn2 "PostgreSQL servisi kurulumdan sonra hazir olmadi - birkac saniye sonra tekrar deneyin." }
            $psqlPath = "psql"
        } catch {
            Write-Warn2 "PostgreSQL otomatik kurulumu basarisiz: $($_.Exception.Message) - elle kurun: https://www.enterprisedb.com/downloads/postgres-postgresql-downloads"
            $psqlPath = $null
        }
    } elseif (-not $psqlPath -and $SkipPgInstall) {
        Write-Warn2 "psql bulunamadi (-SkipPgInstall verildi) - veritabanini elle olusturun: CREATE DATABASE photoai_db;"
    } else {
        Write-Skip "PostgreSQL zaten kurulu ($psqlPath)"
    }

    if ($psqlPath) {
        if (-not $freshInstall) {
            $secure = Read-Host "PostgreSQL 'postgres' kullanicisinin MEVCUT parolasi" -AsSecureString
            $dbPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
                [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
        }
        $env:PGPASSWORD = $dbPassword
        try {
            $exists = & psql -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='photoai_db'" 2>$null
            if ($exists -match "1") {
                Write-Skip "photoai_db zaten var"
            } else {
                & psql -U postgres -c "CREATE DATABASE photoai_db;" | Out-Null
                Write-Ok "photoai_db olusturuldu"
            }
        } catch {
            Write-Warn2 "Veritabani olusturulamadi: $($_.Exception.Message)"
        } finally {
            Remove-Item Env:\PGPASSWORD -ErrorAction SilentlyContinue
        }
    }
}

# --- 4) Qdrant -------------------------------------------------------------
Write-Step 4 "Qdrant"
if ($SkipQdrant) {
    Write-Skip "-SkipQdrant verildi"
} else {
    $qdrantExe = "C:\qdrant\qdrant.exe"
    if (Test-Path $qdrantExe) {
        Write-Skip "C:\qdrant\qdrant.exe zaten var"
    } else {
        try {
            New-Item -ItemType Directory -Force C:\qdrant | Out-Null
            $zipPath = Join-Path $env:TEMP "qdrant.zip"
            $url = "https://github.com/qdrant/qdrant/releases/download/v1.18.1/qdrant-x86_64-pc-windows-msvc.zip"
            Write-Host "  Indiriliyor: $url" -ForegroundColor DarkCyan
            Invoke-WebRequest -Uri $url -OutFile $zipPath
            Expand-Archive -Path $zipPath -DestinationPath C:\qdrant -Force
            Remove-Item $zipPath
            Write-Ok "Qdrant C:\qdrant altina kuruldu"
        } catch {
            Write-Warn2 "Qdrant indirilemedi: $($_.Exception.Message) - elle indirin: https://github.com/qdrant/qdrant/releases"
        }
    }
    Write-Host "  Baslatmak icin (ayri pencerede): cd C:\qdrant; .\qdrant.exe" -ForegroundColor DarkGray
}

# --- 5) Model dosyalari ------------------------------------------------------
Write-Step 5 "Model dosyalari (~1,4 GB)"
if ($SkipModels) {
    Write-Skip "-SkipModels verildi"
} else {
    New-Item -ItemType Directory -Force models\auraface | Out-Null

    $yunet = "models\face_detection_yunet_2023mar.onnx"
    if ((Test-Path $yunet) -and (Get-Item $yunet).Length -gt 100KB) {
        Write-Skip "YuNet zaten mevcut"
    } else {
        try {
            curl.exe -L -o $yunet "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            Write-Ok "YuNet indirildi"
        } catch { Write-Warn2 "YuNet indirilemedi: $($_.Exception.Message)" }
    }

    $aura = "models\auraface\glintr100.onnx"
    if ((Test-Path $aura) -and (Get-Item $aura).Length -gt 100MB) {
        Write-Skip "AuraFace zaten mevcut"
    } else {
        try {
            hf download fal/AuraFace-v1 glintr100.onnx --local-dir models\auraface
            Write-Ok "AuraFace indirildi"
        } catch { Write-Warn2 "AuraFace indirilemedi: $($_.Exception.Message)" }
    }

    $e5 = "models\multilingual-e5-base\model.safetensors"
    if ((Test-Path $e5) -and (Get-Item $e5).Length -gt 500MB) {
        Write-Skip "multilingual-e5-base zaten mevcut"
    } else {
        try {
            hf download intfloat/multilingual-e5-base --local-dir models\multilingual-e5-base
            Write-Ok "multilingual-e5-base indirildi"
        } catch { Write-Warn2 "multilingual-e5-base indirilemedi: $($_.Exception.Message)" }
    }
}

# --- 6) .env -----------------------------------------------------------------
Write-Step 6 ".env"
$envPath = Join-Path $root ".env"
if (Test-Path $envPath) {
    Write-Skip ".env zaten var - UZERINE YAZILMADI"
} else {
    Copy-Item ".env.example" ".env"

    $jwtSecret = python -c "import secrets; print(secrets.token_urlsafe(48))"
    (Get-Content .env) -replace 'JWT_SECRET=your_long_random_secret_here', "JWT_SECRET=$jwtSecret" |
        Set-Content .env -Encoding utf8

    if (-not $dbPassword) {
        $secure = Read-Host "DATABASE_URL icin 'postgres' parolasi" -AsSecureString
        $dbPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
            [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
    }
    (Get-Content .env) -replace 'postgresql://postgres:your_db_password_here@localhost:5432/photoai_db', `
        "postgresql://postgres:$dbPassword@localhost:5432/photoai_db" |
        Set-Content .env -Encoding utf8

    if ($Cpu) {
        Add-Content .env "`nEMBEDDING_DEVICE=cpu"
        Write-Warn2 "CPU modunda EMBEDDING_DEVICE=cpu eklendi - yerel VLM icin GPU gerekir, AI_PROVIDER=bedrock kullanmayi dusunun."
    }

    if ($Lan) {
        $lanOrigin = Read-Host "Frontend'in LAN adresi (ornek: http://192.168.1.50:5173, bos birakilabilir)"
        if ($lanOrigin) {
            (Get-Content .env) -replace 'CORS_ORIGINS=http://localhost:5173', `
                "CORS_ORIGINS=$lanOrigin,http://localhost:5173" |
                Set-Content .env -Encoding utf8
        }
    }

    Write-Ok ".env olusturuldu (DATABASE_URL + JWT_SECRET dolduruldu)"
    Write-Warn2 ".env icinde VLM_BASE_URL/VLM_MODEL/AI_PROVIDER varsayilanlarini (LM Studio) kontrol edin - Bedrock kullanacaksaniz .env'i elle duzenleyin."
}

# --- 7) Alembic migration'lari -------------------------------------------------
Write-Step 7 "Veritabani migration'lari"
try {
    alembic upgrade head
    Write-Ok "alembic upgrade head tamamlandi"
} catch {
    Write-Warn2 "alembic upgrade basarisiz: $($_.Exception.Message) - .env icindeki DATABASE_URL'i kontrol edin."
}

# --- 8) Ilk admin hesabi ---------------------------------------------------
Write-Step 8 "Ilk admin hesabi"
$makeAdmin = Read-Host "Simdi bir admin hesabi olusturulsun mu? (E/h)"
if ($makeAdmin -match '^(e|evet|y)?$') {   # bos (Enter) = evet; -match varsayilan olarak case-insensitive
    $email = Read-Host "  E-posta"
    $secure = Read-Host "  Parola (en az 8 karakter)" -AsSecureString
    $adminPw = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure))
    $displayName = Read-Host "  Ad Soyad"
    try {
        python scripts\create_admin.py --email $email --password $adminPw --display-name $displayName
    } catch {
        Write-Warn2 "Admin olusturulamadi: $($_.Exception.Message)"
    }
} else {
    Write-Skip "elle calistirin: python scripts\create_admin.py --email ... --password ... --display-name ..."
}

# --- Ozet --------------------------------------------------------------------
Write-Host ""
Write-Host "===================== KURULUM TAMAMLANDI =====================" -ForegroundColor Cyan
if ($warnings.Count -gt 0) {
    Write-Host "Uyarilar ($($warnings.Count)):" -ForegroundColor Yellow
    foreach ($w in $warnings) { Write-Host "  - $w" -ForegroundColor Yellow }
    Write-Host ""
}
Write-Host "Servisleri baslatmak icin (4 ayri pencere, ya da bir ust klasordeki ..\start-photoai.bat):" -ForegroundColor White
Write-Host '  uvicorn app.main:app --reload --reload-dir app --port 8001' -ForegroundColor DarkGray
Write-Host '  $env:JOB_TYPES="vlm_analysis"; python -m app.worker.main' -ForegroundColor DarkGray
Write-Host '  $env:JOB_TYPES="face_pipeline"; python -m app.worker.main' -ForegroundColor DarkGray
Write-Host '  $env:JOB_TYPES="semantic_index"; python -m app.worker.main' -ForegroundColor DarkGray
Write-Host ""
Write-Host "Dogrulama: curl.exe http://localhost:8001/health" -ForegroundColor White
Write-Host "Detay/hata cozumu: KURULUM.md" -ForegroundColor White
