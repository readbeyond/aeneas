#!/bin/bash

# __author__ = "Alberto Pettarin"
# __copyright__ = """
#     Copyright 2012-2013, Alberto Pettarin (www.albertopettarin.it)
#     Copyright 2013-2015, ReadBeyond Srl   (www.readbeyond.it)
#     Copyright 2015-2017, Alberto Pettarin (www.albertopettarin.it)
#     """
# __license__ = "GNU AGPL 3"
# __version__ = "1.7.3"
# __email__ = "aeneas@readbeyond.it"
# __status__ = "Production"

# DISABLED
    #DEB="deb-multimedia-keyring_2015.6.1_all.deb"

    # DISABLED
        #echo "[INFO] A.2 Downloading and installing deb-multimedia keyring..."
        #wget -t 5 "http://www.deb-multimedia.org/pool/main/d/deb-multimedia-keyring/$DEB"
        #if ! [ -e "$DEB" ]
        #then
        #    echo "[ERRO] Cannot download the deb-multimedia keyring."
        #    echo "[ERRO] This might be due to a temporary network or server problem."
        #    echo "[ERRO] Please retry installing this Vagrant box later."
        #    echo "[ERRO] If the problem persists, please send an email to aeneas@readbeyond.it"
        #    exit 1
        #fi
        #dpkg -i "$DEB"
        #rm "$DEB"
        #echo "[INFO] A.2 Downloading and installing deb-multimedia keyring... done"

    echo "[INFO] A.1 Adding deb-multimedia to apt sources..."
    sudo sh -c 'echo "deb http://www.deb-multimedia.org jessie main" > /etc/apt/sources.list.d/deb-multimedia.list'
    echo "[INFO] A.1 Adding deb-multimedia to apt sources... done"

    echo "[INFO] A.2 Updating apt..."
    sudo apt-get update
    echo "[INFO] A.2 Updating apt... done"

    echo "[INFO] A.3 Downloading and installing deb-multimedia keyring..."
    sudo apt-get install -y --force-yes deb-multimedia-keyring
    echo "[INFO] A.3 Downloading and installing deb-multimedia keyring... done"

    echo "[INFO] A.4 Updating apt..."
    sudo apt-get update
    echo "[INFO] A.4 Updating apt... done"

    echo "[INFO] B.1 Installing ffmpeg (from deb-multimedia)..."
    sudo apt-get install -y --force-yes ffmpeg
    echo "[INFO] B.1 Installing ffmpeg (from deb-multimedia)... done"

    echo "[INFO] B.2 Installing espeak..."
    sudo apt-get install -y espeak espeak-data libespeak1 libespeak-dev
    echo "[INFO] B.2 Installing espeak... done"

    echo "[INFO] B.3 Installing festival..."
    sudo apt-get install -y festival*
    echo "[INFO] B.3 Installing festival... done"

    echo "[INFO] B.4 Installing common libs using apt-get..."
    sudo apt-get install -y build-essential
    sudo apt-get install -y flac libasound2-dev libsndfile1-dev vorbis-tools
    sudo apt-get install -y libxml2-dev libxslt-dev zlib1g-dev
    sudo apt-get install -y python-dev python-pip
    echo "[INFO] B.4 Installing common libs using apt-get... done"

echo "[INFO] Congratulations, now you can use aeneas!"
