@echo off
rem p4 chain: P4-B (node perturbation control) -> P4-A lr selection -> P4-A main (priority order) -> summarize. sequential, skips finished runs.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\p4\_chain.log
echo ==== p4 chain start %date% %time% >> %LOG%
%PY% -u experiments\p4\run_p4.py npb lr main summarize >> %LOG% 2>&1
echo P4_CHAIN_DONE %date% %time% >> %LOG%
echo done > results\p4\_done
