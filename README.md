This is a bug fix version of https://github.com/CPFL/Autoware/tree/master/ros/src/computing/perception/localization/lib/fast_pcl/ndt_gpu.

CUDA is required to build this program.

---
* add sequance test
```shell
./test_sequance ../../src/autoware/core_perception/ndt_gpu/build/room_scan1.pcd ../../src/autoware/core_perception/ndt_gpu/build/room_scan2.pcd  100
```
* add render flag for node test
```shell
./test_node ../../src/autoware/core_perception/ndt_gpu/build/room_scan1.pcd ../../src/autoware/core_perception/ndt_gpu/build/room_scan2.pcd 1 1
```