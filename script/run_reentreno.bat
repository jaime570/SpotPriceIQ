@echo off
cd /d C:\Users\jaime\Desktop\Jaime_LLorca\proyectos\prediccion-electrica
if not exist logs mkdir logs
set PYTHONIOENCODING=utf-8
set PREFECT_PROFILE=ephemeral
echo ===== %date% %time% ===== >> logs\reentreno.log
C:\Users\jaime\AppData\Local\Programs\Python\Python311\python.exe -m src.orquestacion.reentrenamiento >> logs\reentreno.log 2>&1