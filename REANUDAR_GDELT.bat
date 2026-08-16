@echo off
REM ============================================================
REM  REANUDAR el backfill de GDELT (M5) + su vigia de checkpoints
REM  Uso: doble clic a este archivo. Reanuda desde donde quedo
REM  (salta los dias ya capturados; es seguro ejecutarlo aunque
REM  se hayan capturado mas dias). NO ejecutar si ya esta corriendo.
REM ============================================================
cd /d "C:\Users\snoub\Downloads\NO SE EN QUE GASTAR MI TIRMPO\irfn"

set "PY=.venv\Scripts\python.exe"
set "LOG=data\raw\headlines\_gdelt_run.log"
set "WDLOG=data\raw\headlines\_gdelt_watchdog_run.log"

echo.
echo  Relanzando la captura de GDELT (M5)...
start "GDELT-captura" /min cmd /c "%PY% scripts\capture_headlines.py --max-days 180 > %LOG% 2>&1"

echo  Esperando 5s y relanzando el vigia de checkpoints...
timeout /t 5 /nobreak >nul
start "GDELT-vigia" /min cmd /c "%PY% scripts\gdelt_watchdog.py %LOG% > %WDLOG% 2>&1"

echo.
echo  LISTO. Ambos corriendo en ventanas MINIMIZADAS (en la barra de tareas).
echo  Puedes cerrar ESTA ventana: los procesos siguen vivos.
echo  Para DETENERLOS: cierra las dos ventanas "GDELT-captura" y "GDELT-vigia".
echo.
echo  Ver progreso: abre  data\raw\headlines\_gdelt_watchdog_summary.md
echo.
timeout /t 12
