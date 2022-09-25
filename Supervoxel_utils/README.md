## Supervoxel segmenation

In our method, we adopt the supervoxel segmentation method proposed in [Supervoxel-for-3D-point-clouds](https://github.com/yblin/Supervoxel-for-3D-point-clouds). 
This code can be complied with any complier that supports C++11. 
More details please refer to [Supervoxel-for-3D-point-clouds](https://github.com/yblin/Supervoxel-for-3D-point-clouds).

Compared with the previous code in [Supervoxel-for-3D-point-clouds](https://github.com/yblin/Supervoxel-for-3D-point-clouds), this code makes some modifications: 1) We use the ctypes to call segmentation functions in Python; 2) When segmentation, we determine the desired number of supervoxels in a point cloud.