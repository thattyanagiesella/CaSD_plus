@echo off
echo Starting CaSD Flask app...
cd /d "d:\[1] SAINS DATA\SEMESTER 5/00. RISET DASSBAC\productly-v1.0.0"
if exist venv\Scripts\activate.bat (
  call venv\Scripts\activate.bat
  echo Virtual env activated.
) else (
  echo No venv, using global Python.
)
pip install -r requirements.txt --quiet
black app.py --quiet
python app.py
pause
