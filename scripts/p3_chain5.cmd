@echo off
rem p3 chain 5: waits for chain 4 (results\p3\_done4), then H1 CNN (3 seeds + summary) and the MLP RIA extra. No findstr waits (UTF-8 logs hang it).
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\p3\_chain5.log
echo ==== p3 chain5 waiting for _done4 %date% %time% >> %LOG%
:wait
if exist results\p3\_done4 goto run
ping -n 61 127.0.0.1 >nul
goto wait
:run
echo ==== p3 chain5 start %date% %time% >> %LOG%
%PY% -u experiments\p3\h1cnn.py >> %LOG% 2>&1
%PY% -u experiments\p3\h1c.py extra >> %LOG% 2>&1
echo P3_CHAIN5_DONE %date% %time% >> %LOG%
echo done > results\p3\_done5
