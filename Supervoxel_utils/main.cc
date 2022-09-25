#include "codelibrary/base/log.h"
#include "codelibrary/geometry/io/xyz_io.h"
#include "codelibrary/geometry/point_cloud/pca_estimate_normals.h"
#include "codelibrary/geometry/point_cloud/supervoxel_segmentation.h"
#include "codelibrary/geometry/util/distance_3d.h"
#include "codelibrary/util/tree/kd_tree.h"

#include "vccs_knn_supervoxel.h"
#include "vccs_supervoxel.h"

#include <numeric>

/// Point with Normal.
struct PointWithNormal : cl::RPoint3D {
    PointWithNormal() {}

    cl::RVector3D normal;
};

/**
 * Metric used in VCCS supervoxel segmentation.
 *
 * Reference:
 *   Rusu, R.B., Cousins, S., 2011. 3d is here: Point cloud library (pcl),
 *   IEEE International Conference on Robotics and Automation, pp. 1–4.
 */
class VCCSMetric {
public:
    explicit VCCSMetric(double resolution)
        : resolution_(resolution) {}

    double operator() (const PointWithNormal& p1,
                       const PointWithNormal& p2) const {
        return 1.0 - std::fabs(p1.normal * p2.normal) +
               cl::geometry::Distance(p1, p2) / resolution_ * 0.4;
    }

private:
    double resolution_;
};

/**
 * Save point clouds (with segmentation colors) into the file.
 */
void WritePoints(const char* filename,
                 int n_supervoxels,
                 const cl::Array<cl::RPoint3D>& points,
                 const cl::Array<int>& labels) {
    cl::Array<cl::RGB32Color> colors(points.size());
    std::mt19937 random;
    cl::Array<cl::RGB32Color> supervoxel_colors(n_supervoxels);
    for (int i = 0; i < n_supervoxels; ++i) {
        supervoxel_colors[i] = cl::RGB32Color(random());
    }
    for (int i = 0; i < points.size(); ++i) {
        colors[i] = supervoxel_colors[labels[i]];
    }

    if (cl::geometry::io::WriteXYZPoints(filename, points, colors)) {
        LOG(INFO) << "The points are written into " << filename;
    }

//    system(filename);
}



/**
 * Compute segmentation colors for point clouds.
 */
void ComputeSPColor(int n_supervoxels,
                 const cl::Array<cl::RPoint3D>& points,
                 const cl::Array<int>& labels,
                 int* output_color) {
    cl::Array<cl::RGB32Color> colors(points.size());
    std::mt19937 random;
    cl::Array<cl::RGB32Color> supervoxel_colors(n_supervoxels);
    for (int i = 0; i < n_supervoxels; ++i) {
        supervoxel_colors[i] = cl::RGB32Color(random());
    }
    for (int i = 0; i < points.size(); ++i) {
        colors[i] = supervoxel_colors[labels[i]];
    }
    for (int i = 0; i < points.size(); ++i) {
        output_color[i * 3 + 0] = colors[i].red();
        output_color[i * 3 + 1] = colors[i].green();
        output_color[i * 3 + 2] = colors[i].blue();
    }

//    system(filename);
}


extern "C" int main(float* input_pos, int* input_parameter, int* output_label, int* output_color) {
    LOG_ON(INFO);

    cl::Array<cl::RPoint3D> points;


    int num_points = input_parameter[0];
    int num_supervoxels = input_parameter[1];

    cl::geometry::io::ReadPoints(input_pos, &points, num_points);

    int n_points = points.size();
    // LOG(INFO) << n_points << " points are imported.";

    // LOG(INFO) << "Building KD tree...";
    cl::KDTree<cl::RPoint3D> kdtree;
    kdtree.SwapPoints(&points);

    const int k_neighbors = 15;
    assert(k_neighbors < n_points);

    // LOG(INFO) << "Compute the k-nearest neighbors for each point, and "
    //              "estimate the normal vectors...";

    cl::Array<cl::RVector3D> normals(n_points);
    cl::Array<cl::Array<int> > neighbors(n_points);
    cl::Array<cl::RPoint3D> neighbor_points(k_neighbors);
    for (int i = 0; i < n_points; ++i) {
        kdtree.FindKNearestNeighbors(kdtree.points()[i], k_neighbors,
                                     &neighbors[i]);
        for (int k = 0; k < k_neighbors; ++k) {
            neighbor_points[k] = kdtree.points()[neighbors[i][k]];
        }
        cl::geometry::point_cloud::PCAEstimateNormal(neighbor_points.begin(),
                                                     neighbor_points.end(),
                                                     &normals[i]);
    }
    kdtree.SwapPoints(&points);

    // LOG(INFO) << "Start supervoxel segmentation...";

    cl::Array<PointWithNormal> oriented_points(n_points);
    for (int i = 0; i < n_points; ++i) {
        oriented_points[i].x = points[i].x;
        oriented_points[i].y = points[i].y;
        oriented_points[i].z = points[i].z;
        oriented_points[i].normal = normals[i];
    }

    const double resolution = 1.0;
    VCCSMetric metric(resolution);
    cl::Array<int> labels, supervoxels;
    cl::geometry::point_cloud::SupervoxelSegmentation(oriented_points,
                                                      neighbors,
                                                      resolution,
                                                      num_supervoxels,
                                                      metric,
                                                      &supervoxels,
                                                      &labels);



    for (int i = 0; i < points.size(); ++i) {
        output_label[i] = labels[i];
    }

    ComputeSPColor(num_supervoxels, points, labels, output_color);


    return 0;
}
