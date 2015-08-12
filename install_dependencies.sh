#!/bin/bash

# __author__ = "Alberto Pettarin"
# __copyright__ = """
#     Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
#     Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
#     Copyright 2015,      Alberto Pettarin (www.albertopettarin.it)
#     """
# __license__ = "GNU AGPL 3"
# __version__ = "1.1.0"
# __email__ = "aeneas@readbeyond.it"
# __status__ = "Production"

DEB="deb-multimedia-keyring_2015.6.1_all.deb"

echo "[INFO] A.1 Downloading and installing deb-multimedia keyring..."
wget "http://www.deb-multimedia.org/pool/main/d/deb-multimedia-keyring/$DEB"
dpkg -i "$DEB"
rm "$DEB"
echo "[INFO] A.1 Downloading and installing deb-multimedia keyring... done"

echo "[INFO] A.2 Adding deb-multimedia to apt sources..."
sh -c 'echo "deb http://www.deb-multimedia.org jessie main" >> /etc/apt/sources.list'
echo "[INFO] A.2 Adding deb-multimedia to apt sources... done"

echo "[INFO] A.3 Updating apt..."
apt-get update
echo "[INFO] A.3 Updating apt... done"

echo "[INFO] B.1 Installing common libs using apt-get..."
apt-get install -y build-essential git screen vim htop file unzip
apt-get install -y flac libasound2-dev libsndfile1-dev libxml2-dev libxslt-dev vorbis-tools
apt-get install -y python-beautifulsoup python-dev python-lxml python-numpy python-pip python-scipy
echo "[INFO] B.1 Installing common libs using apt-get... done"

echo "[INFO] B.2 Installing ffmpeg (from deb-multimedia)..."
apt-get install -y ffmpeg
echo "[INFO] B.2 Installing ffmpeg (from deb-multimedia)... done"

echo "[INFO] B.3 Installing espeak..."
apt-get install -y espeak*
echo "[INFO] B.3 Installing espeak... done"

echo "[INFO] C.1 Installing Python modules using pip..."
pip install BeautifulSoup lxml numpy scikits.audiolab
echo "[INFO] C.1 Installing Python modules using pip... done"

echo "[INFO] Congratulations, now you can use aeneas!"
