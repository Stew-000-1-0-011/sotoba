/**
 * @file lib.cpp
 * @author Stew
 * @brief SYCLによる自己位置推定およびLiDARシミュレーション
 * @version 0.1
 * @date 2025-09-22
 * 
 * @copyright Copyright (c) 2025
 * 
 */

/*
- 空間を八分木で切ったほうが計算量は落ちるが、
  まあ地図図形たちは総数が10-20個ほどだろうからこのままで問題なかろう
- unionなり`alignas(Ts_ ...) std::byte [sizeof(std::max({Ts_} ...))]`なりを使うほうが拡張しやすいが、
  やや難解で誰も引き継いでくれなくなる可能性があるため、当該部は泣く泣く削除
*/

#include <cstddef>
#include <limits>
#include <thread>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include "sotoba/cylinder.hpp"
#include "sotoba/exception.hpp"
#include "sotoba/fundamental.hpp"
#include "sotoba/lidar.hpp"
#include "sotoba/rectangle.hpp"


namespace sotoba {
	enum class ShapeType : unsigned char { Null, Rectangle, CylinderOuter };

	// // センサ点のフィルタリング
	// void filter_points_sycl(
	// 	sycl::queue& q,
	// 	sycl::buffer<Vec4>& lidar_points,
	// ) {}

	// SYCLカーネルとして実装された最近接点計算
	void calc_closest_point_sycl(
		sycl::queue& q,
		sycl::buffer<Vec4>& lidar_points_buf,
		sycl::buffer<Rectangle>& rectangles_buf,
		sycl::buffer<CylinderOuter>& cylinder_outers_buf,
		sycl::buffer<Vec4>& closest_points_buf,
		double max_distance
	) {
		q.submit([&](sycl::handler& h) {
			 auto lidar_acc = lidar_points_buf.get_access<sycl::access::mode::read>(h);
			 auto rect_acc = rectangles_buf.get_access<sycl::access::mode::read>(h);
			 auto cyl_acc = cylinder_outers_buf.get_access<sycl::access::mode::read>(h);
			 auto closest_acc = closest_points_buf.get_access<sycl::access::mode::write>(h);

			 h.parallel_for<class CalcClosestPoint>(
				 sycl::range(lidar_points_buf.size()),
				 [=](sycl::id<1> idx) {
					 const Vec4 p = lidar_acc[idx];
					 Vec4 minimum = {Vec3{}, std::numeric_limits<float>::infinity()};

					 for (std::size_t j = 0; j < rect_acc.size(); ++j) {
						 if (const auto ret = rect_acc[j].closest_point_and_distance(p);
							 ret.w() < minimum.w()) {
							 minimum = ret;
						 }
					 }
					 for (std::size_t j = 0; j < cyl_acc.size(); ++j) {
						 if (const auto ret = cyl_acc[j].closest_point_and_distance(p);
							 ret.w() < minimum.w()) {
							 minimum = ret;
						 }
					 }

					 if(minimum.w() > max_distance)
					 {
						minimum = {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
					 }

					 closest_acc[idx] = minimum;
				 }
			 );
		 }).wait_and_throw();
	}

	// 2パス法を採用。桁落ちを防ぐため
	void calc_cross_covariance_sycl(
		sycl::queue& q,
		sycl::buffer<Vec4>& lidar_points_buf,
		sycl::buffer<Vec4>& closest_points_buf,
		Mat3d& cross_covariance,
		Vec4& lidar_centroid,
		Vec4& closest_centroid
	) {
		const std::size_t num_points = lidar_points_buf.size();
		if (num_points == 0) throw LogicException{"num_points == 0"};

		u32 valid_points_count = 0;
		Vec4 lidar_sum{0.0f, 0.0f, 0.0f, 0.0f};
		Vec4 closest_sum{0.0f, 0.0f, 0.0f, 0.0f};

		{
			sycl::buffer<u32> valid_points_count_buf{&valid_points_count, 1};
			sycl::buffer<Vec4> lidar_sum_buf{&lidar_sum, 1};
			sycl::buffer<Vec4> closest_sum_buf{&closest_sum, 1};

			q.submit([&](sycl::handler& h) {
				auto lidar_acc = lidar_points_buf.get_access<sycl::access::mode::read>(h);
				auto closest_acc = closest_points_buf.get_access<sycl::access::mode::read>(h);

				auto count_r = sycl::reduction(valid_points_count_buf, h, sycl::plus<u32>());
				auto lidar_r = sycl::reduction(lidar_sum_buf, h, sycl::plus<Vec4>());
				auto closest_r = sycl::reduction(closest_sum_buf, h, sycl::plus<Vec4>());

				h.parallel_for<class ReductionCalcCentroid>(
					sycl::range(num_points),
					count_r,
					lidar_r,
					closest_r,
					[=](sycl::id<1> idx, auto& count, auto& lidar, auto& closest) {
						if (sycl::isfinite(closest_acc[idx].w())) {
							count.combine(1u);
							lidar += lidar_acc[idx];
							closest += closest_acc[idx];
						}
					}
				);
			});
			q.wait_and_throw();
		}

		if (valid_points_count == 0) throw LogicException{"valid_points_count == 0"};

		lidar_centroid = lidar_sum / static_cast<float>(valid_points_count);
		closest_centroid = closest_sum / static_cast<float>(valid_points_count);
		lidar_centroid.w() = 0.0f;
		closest_centroid.w() = 0.0f;

		Mat3d temp_cross_covariance{};
		{
			sycl::buffer<Mat3d> cross_cov_buf(&temp_cross_covariance, 1);
			q.submit([&](sycl::handler& h) {
				 auto lidar_acc = lidar_points_buf.get_access<sycl::access::mode::read>(h);
				 auto closest_acc = closest_points_buf.get_access<sycl::access::mode::read>(h);

				 auto cov_r = sycl::reduction(cross_cov_buf, h, Mat3d(), sycl::plus<Mat3d>());

				 h.parallel_for<class ReductionAddCrossCovariance>(
					 sycl::range(num_points),
					 cov_r,
					 [=](sycl::id<1> idx, auto& cov) {
						 if (sycl::isfinite(closest_acc[idx].w())) {
							 const Vec3 p = (lidar_acc[idx] - lidar_centroid).xyz();
							 const Vec3 q = (closest_acc[idx] - closest_centroid).xyz();
							 Mat3d partial_cov;
							 partial_cov.v[0] = p.x() * q.x();
							 partial_cov.v[1] = p.x() * q.y();
							 partial_cov.v[2] = p.x() * q.z();
							 partial_cov.v[3] = p.y() * q.x();
							 partial_cov.v[4] = p.y() * q.y();
							 partial_cov.v[5] = p.y() * q.z();
							 partial_cov.v[6] = p.z() * q.x();
							 partial_cov.v[7] = p.z() * q.y();
							 partial_cov.v[8] = p.z() * q.z();
							 cov += partial_cov;
						 }
					 }
				 );
			 }).wait_and_throw();
		}
		cross_covariance = temp_cross_covariance;
	}

	auto calc_pose_error(
		const Mat3d& cross_covariance,
		const Vec4& lidar_centroid,
		const Vec4& closest_centroid
	) -> SE3 {
		Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> H(cross_covariance.v.data());
		Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
		const Eigen::Matrix3f& U = svd.matrixU();
		const Eigen::Matrix3f& V = svd.matrixV();
		Eigen::Matrix3f R = V * U.transpose();

		if (R.determinant() < 0.0f) {
			Eigen::Matrix3f V_prime = V;
			V_prime.col(2) *= -1.0f;
			R = V_prime * U.transpose();
		}

		Eigen::Vector3f lidar_c = {lidar_centroid.x(), lidar_centroid.y(), lidar_centroid.z()};
		Eigen::Vector3f closest_c =
			{closest_centroid.x(), closest_centroid.y(), closest_centroid.z()};
		Eigen::Vector3f t = closest_c - R * lidar_c;

		Eigen::Quaternionf q_eigen(R);
		return SE3{
			.q = {{q_eigen.x(), q_eigen.y(), q_eigen.z(), q_eigen.w()}},
			.p = {t.x(), t.y(), t.z(), 0.0f}
		};
	}

	void calc_local_map_sycl(
		sycl::queue& q,
		const SE3& pose_inverse,
		sycl::buffer<Rectangle>& global_rectangles_buf,
		sycl::buffer<CylinderOuter>& global_cylinder_outers_buf,
		sycl::buffer<Rectangle>& local_rectangles_buf,
		sycl::buffer<CylinderOuter>& local_cylinder_outers_buf
	) {
		using namespace std::chrono_literals;
		// std::this_thread::sleep_for(1ms);
		q.submit([&](sycl::handler& h) {
			 auto global_rect_acc = global_rectangles_buf.get_access<sycl::access::mode::read>(h);
			 auto local_rect_acc = local_rectangles_buf.get_access<sycl::access::mode::write>(h);

			 h.parallel_for<class CalcLocalMapRect>(
				 sycl::range(global_rectangles_buf.size()),
				 [=](sycl::id<1> idx) {
					 Rectangle r = global_rect_acc[idx];
					 r.apply_homogeneous(pose_inverse);
					 local_rect_acc[idx] = r;
				 }
			 );
		 }).wait_and_throw();
		// std::this_thread::sleep_for(1ms);

		q.submit([&](sycl::handler& h) {
			 auto global_cyl_acc =
				 global_cylinder_outers_buf.get_access<sycl::access::mode::read>(h);
			 auto local_cyl_acc =
				 local_cylinder_outers_buf.get_access<sycl::access::mode::write>(h);

			 h.parallel_for<class CalcLocalMapCyl>(
				 sycl::range(global_cylinder_outers_buf.size()),
				 [=](sycl::id<1> idx) {
					 CylinderOuter c = global_cyl_acc[idx];
					 c.apply_homogeneous(pose_inverse);
					 local_cyl_acc[idx] = c;
				 }
			 );
		 }).wait_and_throw();
		// std::this_thread::sleep_for(1ms);
	}
 
	auto icp_self_localization(
		sycl::queue& q,
		const SE3& initial,
		const i32 loop_num,
		sycl::buffer<Vec4>& lidar_points_buf,
		sycl::buffer<Vec4>& closest_points_buf,
		sycl::buffer<Rectangle>& global_rects_buf,
		sycl::buffer<CylinderOuter>& global_cyls_buf,
		sycl::buffer<Rectangle>& local_rects_buf,
		sycl::buffer<CylinderOuter>& local_cyls_buf,
		double max_distance
	) -> SE3 {
		// using namespace std::chrono_literals;

		auto pose = initial;

		for (i32 i = 0; i < loop_num; ++i) {
			// std::this_thread::sleep_for(1ms);

			// 地図を更新
			calc_local_map_sycl(
				q,
				!pose,
				global_rects_buf,
				global_cyls_buf,
				local_rects_buf,
				local_cyls_buf
			);

			// std::this_thread::sleep_for(1ms);

			// SYCLカーネルで最近接点を求める
			calc_closest_point_sycl(
				q,
				lidar_points_buf,
				local_rects_buf,
				local_cyls_buf,
				closest_points_buf,
				max_distance
			);

			// std::this_thread::sleep_for(1ms);

			Mat3d cross_covariance{};
			Vec4 lidar_centroid{};
			Vec4 closest_centroid{};

			// SYCLリダクションで相互共分散行列を求める
			calc_cross_covariance_sycl(
				q,
				lidar_points_buf,
				closest_points_buf,
				cross_covariance,
				lidar_centroid,
				closest_centroid
			);

			// std::this_thread::sleep_for(1ms);

			// SVDで自己位置の差分を求める (ホスト側で実行)
			const SE3 pose_diff =
				calc_pose_error(cross_covariance, lidar_centroid, closest_centroid);

			// poseを更新
			pose = (pose * pose_diff).normalize();
		}

		return pose;
	}

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
	) {
		if (lidar_true_points_buf.size() != lidar.num()) {
			throw LogicException{"lidar_true_points_buf.size() != lidar.num()"};
		}
		if (global_rays_buf.size() != lidar.num()) {
			throw LogicException{"global_rays_buf.size() != lidar.num()"};
		}

		// Rayを取得
		lidar.generate_ray(q, pose, global_rays_buf);

		// 各Rayでの交点を取得
		q.submit([&](sycl::handler& h) {
			 auto global_ray_acc = global_rays_buf.get_access<sycl::access::mode::read>(h);
			 auto global_rect_acc = global_rects_buf.get_access<sycl::access::mode::read>(h);
			 auto global_cyl_acc = global_cyls_buf.get_access<sycl::access::mode::read>(h);
			 auto lidar_point_acc = lidar_true_points_buf.get_access<sycl::access::mode::write>(h);
			 h.parallel_for<class RayCollision>(
				 sycl::range(global_rays_buf.size()),
				 [=](sycl::id<1> idx) {
					 const Ray ray = global_ray_acc[idx];
					 Vec4 minimum = Vec4{Vec3{}, std::numeric_limits<float>::infinity()};

					 for (std::size_t i = 0; i < global_rect_acc.size(); ++i) {
						 const auto ret = global_rect_acc[i].ray_collision(ray);
						 if (ret.w() < minimum.w()) { minimum = ret; }
					 }

					 for (std::size_t i = 0; i < global_cyl_acc.size(); ++i) {
						 const auto ret = global_cyl_acc[i].ray_collision(ray);
						 if (ret.w() < minimum.w()) { minimum = ret; }
					 }

					 lidar_point_acc[idx] = {(!pose).trans(minimum).xyz(), minimum.w()};
				 }
			 );
		 }).wait_and_throw();

		// ノイズを付加
		lidar.add_error(q, seed, lidar_true_points_buf, lidar_observed_points_buf);
	}
} // namespace sotoba
