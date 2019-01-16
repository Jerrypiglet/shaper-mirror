#!/usr/bin/env bash
wget https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip --no-check-certificate
unzip scannet_data_pointnet2.zip
mv data scannet
rm scannet_data_pointnet2.zip
