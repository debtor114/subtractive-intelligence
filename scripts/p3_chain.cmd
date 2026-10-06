@echo off
rem p3 pilot chain: lr select -> H2 (three companies x learner ladder) -> H1 (settlement pruning). sequential, skips finished runs.
cd /d C:\Users\KDI\Downloads\Projects\subtractive-intelligence
set PY=.venv\Scripts\python.exe
set LOG=results\p3\_chain.log
echo ==== p3 chain start %date% %time% >> %LOG%
%PY% -u experiments\p3\run_stage.py lr h2 h1 >> %LOG% 2>&1
echo P3_CHAIN_DONE %date% %time% >> %LOG%
echo done > results\p3\_done
