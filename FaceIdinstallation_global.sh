sudo add-apt-repository ppa:biometrics/bob -y
sudo apt-get update
sudo apt-get install wget pkg-config cmake python-dev python-support python-argparse python-setuptools python-opencv python-numpy python-scipy libblitz1-dev libboost-all-dev libhdf5-serial-dev libtiff4-dev libnetpbm10-dev libpng12-dev libgif-dev libjpeg8-dev libopencv-dev python-matplotlib python-pip python-imaging gcc g++ make -y
sudo apt-get install libffi-dev libssl-dev
pip install requests[security]
pip install numpy
pip install bob.extension
pip install bob.blitz
pip install bob.core
pip install bob.io.base
pip install bob.io.image
pip install bob.ip.color
pip install bob.ip.flandmark
pip install -U scikit-learn

