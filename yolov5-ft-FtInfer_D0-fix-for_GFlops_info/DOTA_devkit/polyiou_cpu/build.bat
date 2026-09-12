cd /d "%~dp0"
del *.pyd
del poly_cpu.cpp
rmdir /s /q build
python setup.py build_ext -i
del poly_cpu.cpp
rmdir /s /q build
