sudo add-apt-repository ppa:biometrics/bob -y
sudo apt-get update
sudo apt-get install wget pkg-config cmake python-dev python-support python-argparse python-setuptools python-opencv python-numpy python-scipy libblitz1-dev libboost-all-dev libhdf5-serial-dev libtiff4-dev libnetpbm10-dev libpng12-dev libgif-dev libjpeg8-dev libopencv-dev python-matplotlib python-pip python-imaging gcc g++ make -y
sudo apt-get install libffi-dev libssl-dev
sudo pip install virtualenv
virtualenv profid
profid/bin/pip install requests[security]
profid/bin/pip install numpy
profid/bin/pip install bob.extension
profid/bin/pip install bob.blitz
profid/bin/pip install bob.core
profid/bin/pip install bob.io.base
profid/bin/pip install bob.io.image
profid/bin/pip install bob.ip.color
profid/bin/pip install bob.ip.flandmark
profid/bin/pip install -U scikit-learn

