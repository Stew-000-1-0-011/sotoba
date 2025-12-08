#pragma once

#include <numbers>

#include "sotoba/math/quaternion.hpp"
#include "sotoba/math/scalar_functions.hpp"
#include "sotoba/math/se3.hpp"
#include "sotoba/math/vec.hpp"

#include "sotoba/math/vec_forward_decl.hpp"
#include "sotoba/stdtypes.hpp"

namespace sotoba::sim::rosetta_lidar_impl {
	using math::UVec3;
	using math::Vec3;
	using math::Vec4;
	using math::SE3;
	namespace quaternion = math::quaternion;
	using std::numbers::pi;

	// 単位はdeg, m, Hz
	struct RosettaLidarConfig final {
		// FOV, 前方を基準とする
		std::pair<float, float> elevation;
		std::pair<float, float> azimuth;
		
		// 誤差
		float distance_base_sigma;
		float k_coeff;
		float angle_sigma;

		// ロゼッタサンプリング
		float omega1;  // 互いに素な回転成分
		float omega2;
		float point_sampling_rate;
		float rotation_speed;
		u32 points_num;
	};

	struct RosettaLidar final {
		u32 points_num;
		float dt;  // s
		float omega1;
		float omega2;
		float el_center;  // rad
		float el_amp;  // rad
		float rotation_speed;  // rad/s
		float distance_base_sigma2;
		float k_coeff2;
		float angle_sigma;

		static auto make(const RosettaLidarConfig& conf) noexcept -> RosettaLidar {
			constexpr float deg_to_rad = pi / 180.f;
			const float el_center = (conf.elevation.first + conf.elevation.second) / 2.f * deg_to_rad;
			const float el_amp = (conf.elevation.second - conf.elevation.first) / 2.f * deg_to_rad;
			
			return RosettaLidar {
				.points_num = conf.points_num
				, .dt = 1.f / conf.point_sampling_rate
				, .omega1 = float(conf.omega1 * 2.f * pi)
				, .omega2 = float(conf.omega2 * 2.f * pi)
				, .el_center = el_center
				, .el_amp = el_amp
				, .rotation_speed = float(conf.rotation_speed * 2.f * pi)
				, .distance_base_sigma2 = math::pow2(conf.distance_base_sigma)
				, .k_coeff2 = math::pow2(conf.k_coeff)
				, .angle_sigma = conf.angle_sigma * deg_to_rad
			};
		}

		// ロゼッタサンプリングでrayを生成
		// i番目のレイを返す
		auto generate_ray(const u32 i, const float t0, auto& normal_distribution_rand_gen) const noexcept -> std::pair<UVec3, UVec3> {
			const float t = t0 + i * this->dt;

			// --- リサージュ/スピログラフ的なアプローチ ---
			// 2つの回転ベクトルの合成として角度を決定

			// 簡易モデル: 
			// 仰角(El)は高速な振動
			// 方位角(Az)は[azimuth_min, azimuth_max]間を周回しながら、さらに少量の振動が加わる
			
			// 1. 垂直方向の振動 (正弦波合成)
			const float el_oscillation = math::sin(this->omega1 * t) * math::cos(this->omega2 * t);
			// -1~1 の範囲をFOVにマッピング
			const float el = this->el_center + el_oscillation * this->el_amp;

			// 2. 水平方向の回転
			// Mid-360は全周スキャン。単純な回転にプリズムのゆらぎが乗るイメージ
			// 回転速度: 例えば 10Hz (適当な値) で一周
			const float az_base = this->rotation_speed * t;
			// ロゼットパターンの特徴である「花びら」を作るために
			// Azにもわずかな振動を加える
			const float az_wobble = math::cos(this->omega1 * t) * 0.1f; 
			const float az = az_base + az_wobble;
			
			// // 角にノイズを加える
			const Vec3 true_rpy = {0.f, -el, az};
			const float noise1 = normal_distribution_rand_gen();
			const float noise2 = normal_distribution_rand_gen();
			const Vec3 noised_rpy = {0.f, -el + this->angle_sigma * noise1, az + this->angle_sigma * noise2};

			const UVec3 true_ray = SE3::rot(quaternion::rpy(true_rpy)).app_uv({1.f, 0.f, 0.f});
			const UVec3 noised_ray = SE3::rot(quaternion::rpy(noised_rpy)).app_uv({1.f, 0.f, 0.f});
			return {true_ray, noised_ray};
		}

		// 真の距離にノイズを加え返す
		auto add_distance_noise(const float distance, const float rho2, auto&& normal_distribution_rand_gen) const noexcept -> float {
			const float cov = this->distance_base_sigma2 + this->k_coeff2 * math::pow2(distance) / rho2;
			return distance + math::sqrt(cov) * normal_distribution_rand_gen();
		}

		auto get_points_num() const noexcept -> u32 {
			return this->points_num;
		}

		auto get_scan_time() const noexcept -> float {
			return this->dt * this->points_num;
		}
	};

	inline constexpr auto mid360(const u32 scan_hz, const float down_sampling_rate = 1.f) -> RosettaLidarConfig {
		return RosettaLidarConfig {
			.elevation = {-2.f, 59.f}
			, .azimuth = {-180.f, 180.f}
			, .distance_base_sigma = 0.03f
			, .k_coeff = 0.0017888543819998318
			, .angle_sigma = 0.15
			, .omega1 = 524'287
			, .omega2 = 6'700'417
			, .point_sampling_rate = 200'000.f * down_sampling_rate
			, .rotation_speed = float(scan_hz)
			, .points_num = u32(200'000 * down_sampling_rate / scan_hz)
		};
	}

	inline constexpr auto sphere(const u32 scan_hz, const float down_sampling_rate = 1.f) -> RosettaLidarConfig {
		return RosettaLidarConfig {
			.elevation = {-90.f, 90.f}
			, .azimuth = {-180.f, 180.f}
			, .distance_base_sigma = 0.03f
			, .k_coeff = 0.0017888543819998318
			, .angle_sigma = 0.15
			, .omega1 = 524'287
			, .omega2 = 6'700'417
			, .point_sampling_rate = 200'000.f * down_sampling_rate
			, .rotation_speed = float(scan_hz)
			, .points_num = u32(200'000 * down_sampling_rate / scan_hz)
		};
	}
}

namespace sotoba::sim {
	using rosetta_lidar_impl::RosettaLidar;
	using rosetta_lidar_impl::RosettaLidarConfig;
	using rosetta_lidar_impl::mid360;
	using rosetta_lidar_impl::sphere;
}