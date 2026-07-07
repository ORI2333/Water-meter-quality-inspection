@echo off
for %%I in ("%~dp0..\..") do set REPO_ROOT=%%~fI
set WORK_DIR=%REPO_ROOT%\data\original_dataset\labelme_center_edit
if not defined PYTHON_EXE set PYTHON_EXE=python
cd /d "%WORK_DIR%"
"%PYTHON_EXE%" -m labelme --nodata --output "%WORK_DIR%" "%WORK_DIR%"
