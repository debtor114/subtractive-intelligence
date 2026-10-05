@echo off
rem p2 chain4: X2 control (finetune at the slow learning rate) then aggregation.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\p2\_chain4.log
echo ==== chain4 start %date% %time% > %LOG%
%PY% -u experiments\p2\x2_control.py >> %LOG% 2>&1
%PY% -u experiments\p2\report_data.py >> %LOG% 2>&1
echo P2_CHAIN4_DONE %date% %time% >> %LOG%
