#include <fast_pcl/ndt_gpu/NormalDistributionsTransform.h>

#include <pcl/point_cloud.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
//#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

#include <time.h>

#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Too few params" << std::endl;
        return -1;
    }

    int n_iter = 1;
    if (argc >= 4)
    {
        n_iter = atoi(argv[3]);
        std::cout << "n_iter: " << n_iter << std::endl;
    }

    // Loading first scan of room.
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
    //pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    clouds.emplace_back(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *clouds.back()) == -1)
    {
        PCL_ERROR("Couldn't read file room_scan1.pcd \n");
        return (-1);
    }
    std::cout << "Loaded " << clouds.back()->size() << " data points from room_scan1.pcd" << std::endl;

    // Loading second scan of room from new perspective.
    //pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    clouds.emplace_back(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[2], *clouds.back()) == -1)
    {
        PCL_ERROR("Couldn't read file room_scan2.pcd \n");
        return (-1);
    }
    std::cout << "Loaded " << clouds.back()->size() << " data points from room_scan2.pcd" << std::endl;

    double total_time = 0;
    auto it = n_iter;
    int target_idx = 0;
    int input_idx = 1;
    while (it--)
    {
        // Filtering input scan to roughly 10% of original size to increase speed of registration.
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
        approximate_voxel_filter.setLeafSize(0.2, 0.2, 0.2);
        approximate_voxel_filter.setInputCloud(clouds[input_idx]);
        approximate_voxel_filter.filter(*filtered_cloud);
        std::cout << "Filtered cloud contains " << filtered_cloud->size() << " data points from room_scan2.pcd" << std::endl;

        gpu::GNormalDistributionsTransform g_ndt;
        // Initializing Normal Distributions Transform (NDT).
        // Setting scale dependent NDT parameters
        // Setting minimum transformation difference for termination condition.
        g_ndt.setTransformationEpsilon(0.01);
        // Setting maximum step size for More-Thuente line search.
        g_ndt.setStepSize(0.1);
        // Setting Resolution of NDT grid structure (VoxelGridCovariance).
        g_ndt.setResolution(1.0);

        // Setting max number of registration iterations.
        g_ndt.setMaximumIterations(35);

        // Setting point cloud to be aligned.
        g_ndt.setInputSource(filtered_cloud);
        // Setting point cloud to be aligned to.
        g_ndt.setInputTarget(clouds[target_idx]);

        // Set initial alignment estimate found using robot odometry.
        Eigen::AngleAxisf init_rotation(0.6931, Eigen::Vector3f::UnitZ());
        Eigen::Translation3f init_translation(1.79387, 0.720047, 0);
        Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();

        // Calculating required rigid transform to align the input cloud to the target cloud.
        pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        clock_t start = clock();
        Eigen::Matrix4f final_trans;
        bool converged = false;
        double fitness_score = -1;

        g_ndt.align(init_guess);
        final_trans = g_ndt.getFinalTransformation();
        converged = g_ndt.hasConverged();
        fitness_score = g_ndt.getFitnessScore();

        clock_t finish = clock();
        double duration = (double)(finish - start) / CLOCKS_PER_SEC;
        total_time += duration;
        std::cout << "Time cost:" << duration << std::endl;

        std::cout << "Normal Distributions Transform has converged:" << converged << " score: " << fitness_score
                << std::endl;

        std::swap(target_idx, input_idx);
    }

    if (n_iter > 0)
    {
        std::cout << "Average time cost:" << total_time / n_iter << std::endl;
    }

    return 0;
}
