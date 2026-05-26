@echo off
echo ========================================
echo   LenScope AI - Starting Servers
echo ========================================
echo.

REM Start backend server in a new window
echo [1/2] Starting Backend Server...
start "LenScope Backend" cmd /k "cd backend && python main.py"

REM Wait for backend to start
timeout /t 5 /nobreak >nul

REM Check if backend is running
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Backend server started successfully on http://localhost:8000
) else (
    echo [ERROR] Backend server failed to start. Check for errors.
    pause
    exit /b 1
)

echo.
echo [2/2] Starting Frontend Server...
start "LenScope Frontend" cmd /k "npm run dev --prefix frontend"

echo.
echo ========================================
echo   Servers Starting...
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000 (or 3001 if 3000 is in use)