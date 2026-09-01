@echo off
REM Master Pipeline Execution Script for MDI3003 Lab 06
REM Student: Lokanth S | Reg No: 23MID0037
echo ======================================================================
echo RUNNING TIME-SERIES CRIME FORECASTING PIPELINE (MDI3003 LAB 06)
echo ======================================================================
python run_all.py
if %ERRORLEVEL% NEQ 0 (
    echo Pipeline execution failed!
    exit /b %ERRORLEVEL%
)
echo Generating Jupyter Notebooks...
python build_notebooks.py
echo Assembling Official 15+ Page DOCX and PDF Reports...
python src/report_generator.py
python -c "import docx2pdf; docx2pdf.convert('23MID0037_Lab06_Report.docx', '23MID0037_Lab06_Report.pdf')"
echo ======================================================================
echo EXECUTION COMPLETED SUCCESSFULLY. ALL ARTIFACTS COMMITTED.
echo ======================================================================
pause
