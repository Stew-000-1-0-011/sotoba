#pragma once

#include <cstddef>

#include "fundamental.hpp"
#include "philox.hpp"

namespace sotoba {
	struct Lidar3d final {
		float vertcal_min;
		float vertcal_max;
		float horizontal_min;
		float horizontal_max;
		float range_precision;
		float angular_precision;

		i32 vertical_num; // レーンの個数
		i32 horizontal_num; // 1レーンあたりの点数

		// グローバルなRayを返す
		void generate_ray(
			this const Lidar3d& self,
			sycl::queue& q,
			const SE3& pose,
			sycl::buffer<Ray>& ray_buf
		) {
			/// @todo_plz: 実装
			q.submit([&](sycl::handler& h) {
				 auto ray_acc = ray_buf.get_access<sycl::access::mode::write>(h);

				 h.parallel_for<class GenerateRays>(sycl::range(self.num()), [=](sycl::id<1> idx) {
					 const i32 v_idx = idx[0] / self.horizontal_num;
					 const i32 h_idx = idx[0] % self.horizontal_num;

					 // 垂直角度を計算
					 const float v_angle = (self.vertical_num > 1) ? self.vertcal_min
							 + (self.vertcal_max - self.vertcal_min) * static_cast<float>(v_idx)
								 / static_cast<float>(self.vertical_num - 1)
																   : self.vertcal_min;

					 // 水平角度を計算
					 const float h_angle = self.horizontal_min
						 + (self.horizontal_max - self.horizontal_min) * static_cast<float>(h_idx)
							 / static_cast<float>(self.horizontal_num);

					 // ローカル座標での方向ベクトルを三角関数で計算
					 const float cos_v = sycl::cos(v_angle);
					 const UVec4 local_dir = {
						 cos_v * sycl::cos(h_angle),
						 cos_v * sycl::sin(h_angle),
						 sycl::sin(v_angle),
						 0.0f
					 };

					 // Lidarの姿勢で方向ベクトルを回転させてグローバルな向きに
					 const UVec4 global_dir = pose.q.rot_vec(local_dir);

					 // レイを生成
					 ray_acc[idx] = Ray{global_dir, pose.p};
				 });
			 }).wait_and_throw();
		}

		// 1フレームあたりの点の総数を返す
		constexpr auto num(this const Lidar3d& self) noexcept -> std::size_t {
			return self.vertical_num * self.horizontal_num;
		}

		/// @todo: ノイズを加えるカーネル書く
		void add_error(
			this const Lidar3d& self,
			sycl::queue& q,
			const u32 seed,
			sycl::buffer<Vec4>& true_points,
			sycl::buffer<Vec4>& observed_points
		) {
			q.submit([&](sycl::handler& h) {
				 auto true_acc = true_points.get_access<sycl::access::mode::read>(h);
				 auto observed_acc = observed_points.get_access<sycl::access::mode::write>(h);
				 h.parallel_for<class AddErrorLidar3d>(
					 sycl::range(true_points.size()),
					 [=](sycl::id<1> idx) {
						 // --- 1. IDとシードから正規分布乱数を3つ生成 ---
						 // 3つの乱数(距離,方位角,仰角)が必要なので、2回生成する
						 u32 id = idx[0];

						 sycl::uint2 r1_uints = philox_2x32(id, seed);
						 sycl::float2 r1_uniforms = uints_to_uniform_floats(r1_uints);
						 sycl::float2 n1_n2 = box_muller_transform(r1_uniforms); // n1, n2 を生成

						 // 2セット目の乱数 (カウンタ値を変えて独立な乱数列を得る)
						 sycl::uint2 r2_uints = philox_2x32(id, seed + 1); // キーを変える
						 sycl::float2 r2_uniforms = uints_to_uniform_floats(r2_uints);
						 sycl::float2 n3_n4 = box_muller_transform(r2_uniforms); // n3, n4 を生成

						 float range_error = n1_n2.x() * self.range_precision;
						 float azimuth_error = n1_n2.y() * self.angular_precision;
						 float elevation_error = n3_n4.x() * self.angular_precision;

						 // --- 2. 誤差の適用 ---
						 const auto p = true_acc[idx];

						 float r = sycl::length(p.xyz());
						 if (r < epsilon) {
							 observed_acc[idx] = p;
							 return;
						 }
						 float azimuth = sycl::atan2(p.y(), p.x());
						 float elevation = sycl::asin(p.z() / r);

						 float noisy_r = r + range_error;
						 float noisy_azimuth = azimuth + azimuth_error;
						 float noisy_elevation = elevation + elevation_error;

						 float x = noisy_r * sycl::cos(noisy_elevation) * sycl::cos(noisy_azimuth);
						 float y = noisy_r * sycl::cos(noisy_elevation) * sycl::sin(noisy_azimuth);
						 float z = noisy_r * sycl::sin(noisy_elevation);

						 observed_acc[idx] = Vec4{x, y, z, 0.f};
					 }
				 );
			 }).wait_and_throw();
		}
	};
} // namespace sotoba

#ifdef sotoba_ENABLE_TESTING
	#include <numbers>
	#include "doctest.h"
	#include "fundamental.hpp"

TEST_SUITE("lidar") {
	using namespace sotoba;
	using namespace sotoba::test;

	TEST_CASE("num") {
		Lidar3d lidar = {
			/*.vertcal_min     =*/0.f,
			/*.vertcal_max     =*/0.f,
			/*.horizontal_min  =*/0.f,
			/*.horizontal_max  =*/0.f,
			/*.range_precision =*/0.f,
			/*angular_precision=*/0.f,
			/*.vertical_num    =*/16,
			/*.horizontal_num  =*/1800,
		};
		CHECK(lidar.num() == 16 * 1800);
	}

	TEST_CASE("generate_ray") {
		// SYCLキューの準備
		sycl::queue q(sycl::default_selector_v);

		// 2x2の4点を出力するLidarを定義
		const Lidar3d lidar = {
			/*.vertcal_min    =*/-static_cast<float>(std::numbers::pi / 4), // -45 deg
			/*.vertcal_max    =*/static_cast<float>(std::numbers::pi / 4), // +45 deg
			/*.horizontal_min =*/-static_cast<float>(std::numbers::pi / 2), // -90 deg
			/*.horizontal_max =*/static_cast<float>(std::numbers::pi / 2), // +90 deg
			/*.range_precision =*/0.f,
			/*angular_precision=*/0.f,
			/*.vertical_num   =*/2,
			/*.horizontal_num =*/2,
		};

		sycl::buffer<Ray> ray_buf(sycl::range(lidar.num()));

		SUBCASE("Identity pose (no transformation)") {
			const SE3 pose = SE3::ide();
			lidar.generate_ray(q, pose, ray_buf);

			sycl::host_accessor host_acc(ray_buf, sycl::read_only);

			// Expected values (local directions are global directions)
			const float v0_angle = -static_cast<float>(std::numbers::pi / 4);
			const float h0_angle = -static_cast<float>(std::numbers::pi / 2);
			const float h1_angle = 0.f; // -PI/2 + PI*1/2

			const Vec4 origin = {0.f, 0.f, 0.f, 0.f};

			// Ray 0: v_idx=0, h_idx=0
			const Ray& r0 = host_acc[0];
			const Vec4 expected_dir0 = {
				sycl::cos(v0_angle) * sycl::cos(h0_angle),
				sycl::cos(v0_angle) * sycl::sin(h0_angle),
				sycl::sin(v0_angle),
				0.f
			};
			CHECK(Check{r0.direction} == Check{expected_dir0});
			CHECK(Check{r0.origin} == Check{origin});

			// Ray 1: v_idx=0, h_idx=1
			const Ray& r1 = host_acc[1];
			const Vec4 expected_dir1 = {
				sycl::cos(v0_angle) * sycl::cos(h1_angle),
				sycl::cos(v0_angle) * sycl::sin(h1_angle),
				sycl::sin(v0_angle),
				0.f
			};
			CHECK(Check{r1.direction} == Check{expected_dir1});
			CHECK(Check{r1.origin} == Check{origin});
		}

		SUBCASE("Translated and rotated pose") {
			// Z軸周りに90度回転し、(10,20,30)へ移動
			const Vec3 rpy = {0.f, 0.f, static_cast<float>(std::numbers::pi / 2)};
			const Vec4 p = {10.f, 20.f, 30.f, 0.f};
			const SE3 pose = {UnitQuaternion::from_rpy(rpy), p};

			lidar.generate_ray(q, pose, ray_buf);

			sycl::host_accessor host_acc(ray_buf, sycl::read_only);

			const float v0_angle = -static_cast<float>(std::numbers::pi / 4);
			const float h1_angle = 0.f;

			// Check Ray 1 (v_idx=0, h_idx=1)
			const Ray& r1 = host_acc[1];
			const Vec4 local_dir1 = {
				sycl::cos(v0_angle) * sycl::cos(h1_angle),
				sycl::cos(v0_angle) * sycl::sin(h1_angle),
				sycl::sin(v0_angle),
				0.f
			};

			// Z軸90度回転 (x, y, z) -> (-y, x, z)
			const Vec4 expected_global_dir1 =
				{-local_dir1.y(), local_dir1.x(), local_dir1.z(), 0.f};

			CHECK(Check{r1.direction} == Check{expected_global_dir1});
			CHECK(Check{r1.origin} == Check{p}); // Origin must be the pose's position
		}
	}

	TEST_CASE("add_error") {
		///@todo: 実装
	}
}
#endif
