@echo off
call venv/Scripts/activate.bat

cd ../tracy

mkdir build
cd build

cmake -G "Unix Makefiles" -DTRACY_STATIC=OFF -DTRACY_CLIENT_PYTHON=ON -DPython_FIND_VIRTUALENV=ONLY ../
make -j 12

cd ../python
python -m pip install wheel setuptools
python setup.py bdist_wheel

echo "Please find the appropriate wheel in $(pwd)/dist and install it with pip"
echo "Example:"
echo "python -m pip install dist/tracy_client-0.10.0-cp311-cp311-macosx_15_0_universal2.whl --force-reinstall"

cd ../..
