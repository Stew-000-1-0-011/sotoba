#include <chrono>
#include <numbers>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "sotoba/fundamental.hpp"
#include "sotoba/exception.hpp"
#include "sotoba/rectangle.hpp"
#include "sotoba/cylinder.hpp"
#include "sotoba/lidar.hpp"
#include "sotoba/lib.hpp"


TEST_SUITE("lib") {
	using namespace sotoba;
	using namespace sotoba::test;

	TEST_CASE("calc_closest_point_sycl") {
		sycl::queue q;
		std::vector<Vec4> lidar_points = {
			{1.f, 1.f, 1.f, 0.f}, // 長方形の面の内側(Z=5)の射影
			{3.f, 0.f, 0.f, 0.f}, // 長方形の面の外側
			{7.f, 0.f, 6.f, 0.f}, // 円柱の側面に近い
		};
		// XY平面、中心(0,0,5)、大きさ(4x5)の長方形
		std::vector<Rectangle> rectangles = {
			Rectangle{
				.center={0.f, 0.f, 5.f, 0.f}, 
				.u_axis_and_hlen={1.f, 0.f, 0.f, 2.f}, 
				.v_axis_and_hlen={0.f, 1.f, 0.f, 2.5f}, // 3.0fから2.5fに変更 
				.normal={0.f, 0.f, -1.f, 0.f}
			}
		};
		// Z軸に平行、中心(10,0,5)、半径2、高さ4(-2~2)の円柱
		std::vector<CylinderOuter> cylinders = {CylinderOuter{
			.center_and_radius = {10.f, 0.f, 5.f, 2.f},
			.axis_and_hheight = {0.f, 0.f, 1.f, 2.f}
		}};
		std::vector<Vec4> closest_points(lidar_points.size());

		sycl::buffer<Vec4> lidar_buf(lidar_points);
		sycl::buffer<Rectangle> rect_buf(rectangles);
		sycl::buffer<CylinderOuter> cyl_buf(cylinders);
		sycl::buffer<Vec4> closest_buf(closest_points);

		calc_closest_point_sycl(q, lidar_buf, rect_buf, cyl_buf, closest_buf);
		q.wait_and_throw();
		auto acc = closest_buf.get_host_access();

		// 1. {1,1,1} -> 長方形の{1,1,5}が最近接点。距離の2乗は(5-1)^2 = 16
		CHECK(Check{acc[0]} == Check{Vec4{1.f, 1.f, 5.f, 16.f}});

		// 2. {3,0,0} -> 長方形の端{2,0,5}が最近接点。距離の2乗は(3-2)^2 + (5-0)^2 = 1+25=26
		CHECK(Check{acc[1]} == Check{Vec4{2.f, 0.f, 5.f, 26.f}});

		// 3. {7,0,6} -> 円柱の側面{8,0,6}が最近接点。距離の2乗は(8-7)^2 = 1
		CHECK(Check{acc[2]} == Check{Vec4{8.f, 0.f, 6.f, 1.f}});
	}

	TEST_CASE("calc_cross_covariance_sycl") {
		sycl::queue q{sycl::cpu_selector_v};

		SUBCASE("valid points") {
			std::vector<Vec4> lidar_points = {{1.0f, 0.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f, 0.0f}};
			std::vector<Vec4> closest_points = {
				{2.0f, 0.0f, 0.0f, 1.f},
				{-2.0f, 0.0f, 0.0f, 1.f}
			}; // wは有限値なら何でも良い
			sycl::buffer<Vec4> lidar_buf(lidar_points);
			sycl::buffer<Vec4> closest_buf(closest_points);
			Mat3d cross_covariance;
			Vec4 lidar_centroid, closest_centroid;

			calc_cross_covariance_sycl(
				q,
				lidar_buf,
				closest_buf,
				cross_covariance,
				lidar_centroid,
				closest_centroid
			);

			CHECK(Check{lidar_centroid} == Check{Vec4{0.0f, 0.0f, 0.0f, 0.0f}});
			CHECK(Check{closest_centroid} == Check{Vec4{0.0f, 0.0f, 0.0f, 0.0f}});
			CHECK(doctest::Approx(cross_covariance.v[0]) == 4.0f); // (1*2) + (-1*-2)
			for (i32 i = 1; i < 9; ++i) CHECK(doctest::Approx(cross_covariance.v[i]) == 0.0f);
		}

		SUBCASE("no valid points throws exception") {
			std::vector<Vec4> lidar_points = {{1.0f, 0.0f, 0.0f, 0.0f}};
			std::vector<Vec4> closest_points = {
				{0.0f, 0.0f, 0.0f, std::numeric_limits<float>::infinity()}
			};
			sycl::buffer<Vec4> lidar_buf(lidar_points);
			sycl::buffer<Vec4> closest_buf(closest_points);
			Mat3d cross_covariance;
			Vec4 lidar_centroid, closest_centroid;

			CHECK_THROWS_AS(
				calc_cross_covariance_sycl(
					q,
					lidar_buf,
					closest_buf,
					cross_covariance,
					lidar_centroid,
					closest_centroid
				),
				const LogicException&
			);
		}
	}

	TEST_CASE("calc_pose_error") {
		SUBCASE("translation only") {
			Mat3d cross_covariance{}; // H = 0
			Vec4 lidar_centroid = {1.0f, 2.0f, 3.0f, 0.0f};
			Vec4 closest_centroid = {1.5f, 2.5f, 3.5f, 0.0f};
			SE3 pose_error = calc_pose_error(cross_covariance, lidar_centroid, closest_centroid);

			CHECK(Check{pose_error.q} == Check{UnitQuaternion::one()});
			CHECK(Check{pose_error.p} == Check{Vec4{0.5f, 0.5f, 0.5f, 0.0f}});
		}

		SUBCASE("rotation only") {
			Mat3d H;
			H.v = {0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
			Vec4 lidar_centroid = {0.0f, 0.0f, 0.0f, 0.0f};
			Vec4 closest_centroid = {0.0f, 0.0f, 0.0f, 0.0f};
			SE3 pose_error = calc_pose_error(H, lidar_centroid, closest_centroid);

			const auto expected_q =
				UnitQuaternion::from_rpy({0.f, 0.f, static_cast<float>(std::numbers::pi / 2.0)});
			CHECK(Check{pose_error.q} == Check{expected_q});
			CHECK(Check{pose_error.p} == Check{Vec4{0.0f, 0.0f, 0.0f, 0.0f}});
		}
	}

	TEST_CASE("Integration Test") {
		sycl::queue q{sycl::cpu_selector_v};
		std::vector<Rectangle> global_rects = {
			{.center = {0.f, 0.f, -5.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .normal = {0.f, 0.f, 1.f, 0.f}}, // Floor
			{.center = {-5.f, 0.f, 0.f, 0.f},
			 .u_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {1.f, 0.f, 0.f, 0.f}}, // LeftWall
			 {.center = {0.f, -5.f, 0.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {0.f, 1.f, 0.f, 0.f}}, // BackWall
			 {.center = {0.f, 0.f, -5.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .normal = {0.f, 0.f, 1.f, 0.f}}, // Floor
			{.center = {-5.f, 0.f, 0.f, 0.f},
			 .u_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {1.f, 0.f, 0.f, 0.f}}, // LeftWall
			 {.center = {0.f, -5.f, 0.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {0.f, 1.f, 0.f, 0.f}}, // BackWall
			 {.center = {0.f, 0.f, -5.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .normal = {0.f, 0.f, 1.f, 0.f}}, // Floor
			{.center = {-5.f, 0.f, 0.f, 0.f},
			 .u_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {1.f, 0.f, 0.f, 0.f}}, // LeftWall
			 {.center = {0.f, -5.f, 0.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {0.f, 1.f, 0.f, 0.f}}, // BackWall
			 {.center = {0.f, 0.f, -5.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .normal = {0.f, 0.f, 1.f, 0.f}}, // Floor
			{.center = {-5.f, 0.f, 0.f, 0.f},
			 .u_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {1.f, 0.f, 0.f, 0.f}}, // LeftWall
			 {.center = {0.f, -5.f, 0.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {0.f, 1.f, 0.f, 0.f}}, // BackWall
			 {.center = {0.f, 0.f, -5.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .normal = {0.f, 0.f, 1.f, 0.f}}, // Floor
			{.center = {-5.f, 0.f, 0.f, 0.f},
			 .u_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {1.f, 0.f, 0.f, 0.f}}, // LeftWall
			 {.center = {0.f, -5.f, 0.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {0.f, 1.f, 0.f, 0.f}}, // BackWall
			 {.center = {0.f, 0.f, -5.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .normal = {0.f, 0.f, 1.f, 0.f}}, // Floor
			{.center = {-5.f, 0.f, 0.f, 0.f},
			 .u_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {1.f, 0.f, 0.f, 0.f}}, // LeftWall
			 {.center = {0.f, -5.f, 0.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {0.f, 1.f, 0.f, 0.f}}, // BackWall
			 {.center = {0.f, 0.f, -5.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .normal = {0.f, 0.f, 1.f, 0.f}}, // Floor
			{.center = {-5.f, 0.f, 0.f, 0.f},
			 .u_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {1.f, 0.f, 0.f, 0.f}}, // LeftWall
			 {.center = {0.f, -5.f, 0.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {0.f, 1.f, 0.f, 0.f}}, // BackWall
			 {.center = {0.f, 0.f, -5.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .normal = {0.f, 0.f, 1.f, 0.f}}, // Floor
			{.center = {-5.f, 0.f, 0.f, 0.f},
			 .u_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {1.f, 0.f, 0.f, 0.f}}, // LeftWall
			 {.center = {0.f, -5.f, 0.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {0.f, 1.f, 0.f, 0.f}}, // BackWall
			 {.center = {0.f, 0.f, -5.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .normal = {0.f, 0.f, 1.f, 0.f}}, // Floor
			{.center = {-5.f, 0.f, 0.f, 0.f},
			 .u_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {1.f, 0.f, 0.f, 0.f}}, // LeftWall
			 {.center = {0.f, -5.f, 0.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {0.f, 1.f, 0.f, 0.f}}, // BackWall
			 {.center = {0.f, 0.f, -5.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .normal = {0.f, 0.f, 1.f, 0.f}}, // Floor
			{.center = {-5.f, 0.f, 0.f, 0.f},
			 .u_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {1.f, 0.f, 0.f, 0.f}}, // LeftWall
			 {.center = {0.f, -5.f, 0.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {0.f, 1.f, 0.f, 0.f}}, // BackWall
			 {.center = {0.f, 0.f, -5.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .normal = {0.f, 0.f, 1.f, 0.f}}, // Floor
			{.center = {-5.f, 0.f, 0.f, 0.f},
			 .u_axis_and_hlen = {0.f, 1.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {1.f, 0.f, 0.f, 0.f}}, // LeftWall
			 {.center = {0.f, -5.f, 0.f, 0.f},
			 .u_axis_and_hlen = {1.f, 0.f, 0.f, 5.f},
			 .v_axis_and_hlen = {0.f, 0.f, 1.f, 5.f},
			 .normal = {0.f, 1.f, 0.f, 0.f}}, // BackWall
		};
		std::vector<CylinderOuter> global_cyls {
			{.center_and_radius = {0.f, 0.f, 0.f, 0.f},
			 .axis_and_hheight = {0.f, 0.f, 1.f, 0.f}}
		};

		SE3 true_pose = SE3{UnitQuaternion::one(), {0.0f, 0.0f, 0.0f, 0.0f}}.normalize();

		const Lidar3d lidar = {
			.vertcal_min = -std::numbers::pi,
			.vertcal_max = std::numbers::pi,
			.horizontal_min = 0.f,
			.horizontal_max = static_cast<float>(2.0 * std::numbers::pi),
			.range_precision = 0.02f,
			.angular_precision = 0.15f / 180.f * std::numbers::pi_v<float>,
			.vertical_num = 10,
			.horizontal_num = 200,
		};
		std::vector<Vec4> lidar_points(lidar.num());
		std::vector<Ray> rays(lidar.num());

		sycl::buffer<Rectangle> global_rects_buf(global_rects);
		sycl::buffer<CylinderOuter> global_cyls_buf(global_cyls);
		sycl::buffer<Vec4> lidar_true_points_buf(lidar_points.data(), lidar.num());
		sycl::buffer<Vec4> lidar_observed_points_buf(lidar_points.data(), lidar.num());
		sycl::buffer<Ray> rays_buf(rays.data(), lidar.num());
		simulate_lidar(
			q,
			lidar,
			true_pose,
			0,
			global_rects_buf,
			global_cyls_buf,
			rays_buf,
			lidar_true_points_buf,
			lidar_observed_points_buf
		);

		SE3 initial_pose = true_pose
			* SE3{UnitQuaternion::from_rpy({0.1f, 0.1f, 0.1f}), {0.5f, 0.5f, 0.5f, 0.0f}};
		initial_pose.normalize();

		std::vector<Vec4> closest_points(lidar.num());
		std::vector<Rectangle> local_rects(global_rects.size());
		std::vector<CylinderOuter> local_cyls(global_cyls.size());

		sycl::buffer<Vec4> closest_points_buf(closest_points);
		sycl::buffer<Rectangle> local_rects_buf(local_rects);
		sycl::buffer<CylinderOuter> local_cyls_buf(local_cyls);

		{
			SE3 estimated_pose{};
			const auto start = std::chrono::system_clock::now();
			const i32 loop_num = 20;
			const i32 measure_time_loop_num = 100;
			for(i32 i = 0; i < measure_time_loop_num; ++i) {
				[&]() noexcept {
					estimated_pose = icp_self_localization(
						q,
						initial_pose,
						loop_num,
						lidar_observed_points_buf,
						closest_points_buf,
						global_rects_buf,
						global_cyls_buf,
						local_rects_buf,
						local_cyls_buf
					);
				}();
			}
			const auto end = std::chrono::system_clock::now();
			std::cout << "time is " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() / measure_time_loop_num << std::endl;
			CHECK(Check{estimated_pose, 0.01f} == Check{true_pose, 0.01f});
			std::cout << "estimated is " << Check{estimated_pose} << std::endl;
		}
	}
}
