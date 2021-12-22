#!/usr/bin/env bash

echo "Downloading BigGAN weights"
wget -P generators http://ganalyze.csail.mit.edu/models/biggan-128.pth
wget -P generators http://ganalyze.csail.mit.edu/models/biggan-256.pth
wget -P generators http://ganalyze.csail.mit.edu/models/biggan-512.pth