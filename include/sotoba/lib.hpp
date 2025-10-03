#pragma once

#include <limits>
#include "cylinder.hpp"
#include "fundamental.hpp"
#include "lidar.hpp"
#include "rectangle.hpp"

namespace sotoba {
	// SYCLカーネルとして実装された最近接点計算
	void calc_closest_point_sycl(
		sycl::queue& q,
		sycl::buffer<Vec4>& lidar_points_buf,
		sycl::buffer<Rectangle>& rectangles_buf,
		sycl::buffer<CylinderOuter>& cylinder_outers_buf,
		sycl::buffer<Vec4>& closest_points_buf
	);

	// 2パス法を採用。桁落ちを防ぐため
	void calc_cross_covariance_sycl(
		sycl::queue& q,
		sycl::buffer<Vec4>& lidar_points_buf,
		sycl::buffer<Vec4>& closest_points_buf,
		Mat3d& cross_covariance,
		Vec4& lidar_centroid,
		Vec4& closest_centroid
	);

	auto calc_pose_error(
		const Mat3d& cross_covariance,
		const Vec4& lidar_centroid,
		const Vec4& closest_centroid
	) -> SE3;

	void calc_local_map_sycl(
		sycl::queue& q,
		const SE3& pose_inverse,
		sycl::buffer<Rectangle>& global_rectangles_buf,
		sycl::buffer<CylinderOuter>& global_cylinder_outers_buf,
		sycl::buffer<Rectangle>& local_rectangles_buf,
		sycl::buffer<CylinderOuter>& local_cylinder_outers_buf
	);

	auto icp_self_localization(
		sycl::queue& q,
		const SE3& initial,
		const i32 loop_num,
		sycl::buffer<Vec4>& lidar_points_buf,
		sycl::buffer<Vec4>& closest_points_buf,
		sycl::buffer<Rectangle>& global_rects_buf,
		sycl::buffer<CylinderOuter>& global_cyls_buf,
		sycl::buffer<Rectangle>& local_rects_buf,
		sycl::buffer<CylinderOuter>& local_cyls_buf
	) -> SE3;

	void simulate_lidar(
		sycl::queue& q,
		const Lidar3d& lidar,
		const SE3& pose,
		const u32 seed,
		sycl::buffer<Rectangle>& global_rects_buf,
		sycl::buffer<CylinderOuter>& global_cyls_buf,
		sycl::buffer<Ray>& global_rays_buf,
		sycl::buffer<Vec4>& lidar_true_points_buf,
		sycl::buffer<Vec4>& lidar_observed_points_buf
	);

} // namespace sotoba
