@echo off
rem p4 chain 2: waits for chain 1 (results\p4\_done), then P4-B2 and a fresh summary. No findstr waits.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\p4\_chain2.log
echo ==== p4 chain2 waiting for _done %date% %time% >> %LOG%
:wait
if exist results\p4\_done goto run
ping -n 61 127.0.0.1 >nul
goto wait
:run
echo ==== p4 chain2 start %date% %time% >> %LOG%
%PY% -u experiments\p4\run_p4.py npb2 summarize >> %LOG% 2>&1
echo P4_CHAIN2_DONE %date% %time% >> %LOG%
echo done > results\p4\_done2
