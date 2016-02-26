#!/bin/sh
apt-get install git -y
apt-get install python3 -y
echo "Type your bitbucket username, followed by [ENTER]:"
read username
url = "https://"
url += $username
url += "@bitbucket.org/"
url += $username
url += "/chem.git"
git clone "$url"
cd chem
python3 setup.py install

