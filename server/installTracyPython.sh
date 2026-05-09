source venv/bin/activate

cd ../tracy

mkdir build
cd build

cmake -DTRACY_STATIC=OFF -DTRACY_CLIENT_PYTHON=ON -DPython_EXECUTABLE=$(which python) -DPython_FIND_VIRTUALENV=ONLY ../
make -j $(nproc)

cd ../python
python -m pip install wheel setuptools
python setup.py bdist_wheel

echo "Please find the appropriate wheel in $(pwd)/dist and install it with pip"
echo "Example:"
echo "python3 -m pip install dist/tracy_client-0.10.0-cp311-cp311-macosx_15_0_universal2.whl --force-reinstall"
