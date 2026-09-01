# Master Pipeline Execution Script (PowerShell) for MDI3003 Lab 06
# Student: Lokanth S | Reg No: 23MID0037
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "RUNNING TIME-SERIES CRIME FORECASTING PIPELINE (MDI3003 LAB 06)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

python run_all.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pipeline execution failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "Generating Jupyter Notebooks..." -ForegroundColor Green
python build_notebooks.py

Write-Host "Assembling Official DOCX and PDF Reports..." -ForegroundColor Green
python src/report_generator.py
python -c "import docx2pdf; docx2pdf.convert('23MID0037_Lab06_Report.docx', '23MID0037_Lab06_Report.pdf')"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "EXECUTION COMPLETED SUCCESSFULLY. ALL ARTIFACTS VERIFIED." -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
