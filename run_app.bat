@echo off
cd /d "%~dp0"
echo.
echo  ========================================
echo   SB30 부하 해리력 분석 시스템 v1.0
echo  ========================================
echo.
echo  접속 주소 : http://localhost:8502
echo  종료하려면 이 창을 닫으세요.
echo.

:: 기존 8502 포트 사용 프로세스 종료
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8502 " ^| findstr "LISTENING"') do (
    echo  [기존 프로세스 종료] PID %%a
    taskkill /PID %%a /F >nul 2>&1
)

streamlit run app.py ^
    --server.port 8502 ^
    --browser.serverAddress localhost ^
    --browser.gatherUsageStats false

pause
