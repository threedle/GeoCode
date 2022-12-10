mkdir ~/Blender
cd ~/Blender
wget https://mirror.freedif.org/blender/release/Blender3.2/blender-3.2.0-linux-x64.tar.xz
tar -xvf blender-3.2.0-linux-x64.tar.xz
BLENDER_PYTHON_BIN=~/Blender/blender-3.2.0-linux-x64/3.2/python/bin
cd ${BLENDER_PYTHON_BIN}
wget -P /tmp https://bootstrap.pypa.io/get-pip.py
${BLENDER_PYTHON_BIN}/python3.10 /tmp/get-pip.py
${BLENDER_PYTHON_BIN}/python3.10 -m pip install --upgrade pip
${BLENDER_PYTHON_BIN}/python3.10 -m pip install pyyaml
${BLENDER_PYTHON_BIN}/python3.10 -m pip install tqdm