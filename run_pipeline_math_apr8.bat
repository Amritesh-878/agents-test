@echo off
set FFMPEG_DIR=C:\Users\ansh\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin
set CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
set PATH=%FFMPEG_DIR%;%CUDA_DIR%;%PATH%
cd /d "C:\Users\ansh\Desktop\ISL\Agents_test"
echo Starting pipeline... output goes to output\pipeline_run.log
.venv\Scripts\python -m scripts.run_pipeline ^
  --input "C:\Users\ansh\Desktop\ISL\Agents_test\Videos\Math.01_A _AY2025-26_Linear Equation Scaffolding Time and Work_05_08 Apr.zip" ^
  --output-dir output/ ^
  --teacher "Nisha" ^
  --attendance "C:\Users\ansh\Desktop\ISL\Agents_test\attendance\participants_84538437552_2026_04_05.csv" ^
  --db-url "postgresql://postgres:1077@localhost:5432/adira" ^
  > output\pipeline_run.log 2>&1
echo.
echo Done! Check output\pipeline_run.log for results.
pause
