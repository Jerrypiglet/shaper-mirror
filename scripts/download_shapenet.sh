#!/usr/bin/env bash
# raw data
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
rm shapenetcore_partanno_segmentation_benchmark_v0.zip
mv shapenetcore_partanno_segmentation_benchmark_v0 shapenet

# pointnet
wget https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip --no-check-certificate
unzip shapenet_part_seg_hdf5_data.zip
rm shapenet_part_seg_hdf5_data.zip
mv hdf5_data shapenet_hdf5

# pointnet++
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
rm shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
mv shapenetcore_partanno_segmentation_benchmark_v0_normal shapenet_normal