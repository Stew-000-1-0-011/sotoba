#include <chrono>
#include <concepts>
#include <limits>
#include <numbers>
#include <print>
#include <random>
#include <thread>
#include <variant>

#include "sotoba/icp_resource/normal_known_icp.hpp"
#include "sotoba/icp_resource/resource.hpp"

#include "sotoba/math/quaternion.hpp"
#include "sotoba/math/scalar_functions.hpp"
#include "sotoba/math/se3.hpp"
#include "sotoba/math/vec.hpp"

#include "sotoba/random/xoshiro256pp.hpp"

#include "sotoba/repr.hpp"
#include "sotoba/sim/lap_timer.hpp"
#include "sotoba/sim/lidar2d.hpp"

#include "sotoba/surface/box.hpp"
#include "sotoba/surface/rectangle.hpp"

#include "../pango.hpp"

using namespace sotoba;
using namespace math;
constexpr float pi = std::numbers::pi;
using namespace std::chrono_literals;

#ifndef sotoba_USE_SYCL
using Vec6 = Vec<6>;
#else
using Vec6 = Vec<8>;
#endif

int main() {
	using Variant = std::variant<surface::BoxOuter, surface::Rectangle>;

	u32 loop_num = 10;
	float accept_distance = 0.06;
	float scan_hz = 10.f;
	float trans_speed = 4.0f;
	float rot_speed = 1.5 * pi;
	Vec6 tikhnov{};
	std::cin >> loop_num >> accept_distance >> scan_hz;
	std::cin >> trans_speed >> rot_speed;
	std::cin >> tikhnov[0] >> tikhnov[1] >> tikhnov[2] >> tikhnov[3] >> tikhnov[4] >> tikhnov[5];

	// オブジェクト作成
	// データの実体グループ1: 静的な環境（壁や床など）
	std::vector<Variant> environment_storage{};
	environment_storage.reserve(2);

	// // 1. BoxOuter: 原点にある正当な箱 (回転なし、全壁あり)
	// environment_storage.emplace_back(surface::BoxOuter(
	// 	Vec3{0.0f, 0.0f, 0.0f},  // center
	// 	SquareMat<3>::ide(),  // rot
	// 	Vec3{10.0f, 10.0f, 10.0f},   // hlens (ハーフサイズ)
	// 	std::array<bool, 6>{false, false, false, false, false, false} // 全ての壁が存在
	// ));

	// // 2. Rectangle: 床面 (Y = -5.0f, 上向き法線)
	// environment_storage.emplace_back(surface::Rectangle(
	// 	Vec3{0.0f, -5.0f, 0.0f},     // center
	// 	Vec4{1.0f, 0.0f, 0.0f, 20.0f}, // u_axis (X軸) + half_u_len
	// 	Vec4{0.0f, 0.0f, 1.0f, 20.0f}, // v_axis (Z軸) + half_v_len
	// 	UVec3{0.0f, 1.0f, 0.0f}      // normal (Y軸)
	// ));

	// 外側を囲う大きな囲い
	environment_storage.emplace_back(
		surface::Rectangle(
			Vec3{0.f, 0.f, -2.f},
			Vec4{1.f, 0.f, 0.f, 2.f},
			Vec4{0.f, 1.f, 0.f, 2.f},
			UVec3{0.f, 0.f, 1.f}
		)
	);
	environment_storage.emplace_back(
		surface::Rectangle(
			Vec3{0.f, 0.f, 2.f},
			Vec4{1.f, 0.f, 0.f, 2.f},
			Vec4{0.f, 1.f, 0.f, 2.f},
			UVec3{0.f, 0.f, -1.f}
		)
	);
	environment_storage.emplace_back(
		surface::Rectangle(
			Vec3{-2.f, 0.f, 0.f},
			Vec4{0.f, 1.f, 0.f, 2.f},
			Vec4{0.f, 0.f, 1.f, 2.f},
			UVec3{1.f, 0.f, 0.f}
		)
	);
	environment_storage.emplace_back(
		surface::Rectangle(
			Vec3{2.f, 0.f, 0.f},
			Vec4{0.f, 1.f, 0.f, 2.f},
			Vec4{0.f, 0.f, 1.f, 2.f},
			UVec3{-1.f, 0.f, 0.f}
		)
	);
	environment_storage.emplace_back(
		surface::Rectangle(
			Vec3{0.f, -2.f, 0.f},
			Vec4{0.f, 0.f, 1.f, 2.f},
			Vec4{1.f, 0.f, 0.f, 2.f},
			UVec3{0.f, 1.f, 0.f}
		)
	);
	environment_storage.emplace_back(
		surface::Rectangle(
			Vec3{0.f, 2.f, 0.f},
			Vec4{0.f, 0.f, 1.f, 2.f},
			Vec4{1.f, 0.f, 0.f, 2.f},
			UVec3{0.f, -1.f, 0.f}
		)
	);

	// // データの実体グループ2: 動的な障害物や別のエリア
	// std::vector<Variant> obstacles_storage;
	// obstacles_storage.reserve(2);

	// // 3. BoxOuter: 少し回転し、一部の壁がない箱
	// environment_storage.emplace_back(surface::BoxOuter(
	// 	Vec3{0.0f, 0.0f, 0.0f},  // center
	// 	quaternion::to_mat(quaternion::ypr({0.f, 0.f, pi / 4})),  // rot
	// 	Vec3{2.0f, 4.0f, 2.0f},  // hlens
	// 	std::array<bool, 6>{true, false, true, false, false, false} // 一部の壁(0番, 2番)が存在しない
	// ));

	// // 4. Rectangle: 斜めの板
	// // 法線を(sqrt(2), sqrt(2), 0)のように設定
	// obstacles_storage.emplace_back(surface::Rectangle(
	// 	Vec3{0.0f, 0.0f, 0.0f},     // center
	// 	Vec4{math::sqrt(2), -math::sqrt(2), 0.0f, 5.0f},   // u_axis (傾いた軸)
	// 	Vec4{0.0f, 0.0f, 1.0f, 5.0f}, // v_axis (Z軸はそのまま)
	// 	UVec3{math::sqrt(2), math::sqrt(2), 0.0f}          // normal
	// ));

	const std::vector<std::span<const Variant>> objects = {
		std::span{environment_storage}
		// , std::span{obstacles_storage}
	};

	// LiDAR作成
	const auto lidar = sim::Lidar2d::make(sim::utm_30lx(scan_hz));
	// 乱数生成器
	auto rand_gen_u64 = sotoba::random::Xoshiro256pp{0};
	std::normal_distribution<float> normal_dist{0.f, 1.f};
	auto rand_gen = [&]() { return normal_dist(rand_gen_u64); };
	// ライダー時刻
	float t = 0.f;

	// 各オブジェクトの真の姿勢
	std::vector<SE3> true_poses(objects.size());
	true_poses[0] = SE3::ide();
	// true_poses[0] = SE3::trans({0.f, 20.f, 0.f});
	// true_poses[1] = SE3::trans({5.f, 10.f, 0.f});
	// 各オブジェクトの推定姿勢
	std::vector<SE3> estimated_poses = true_poses;
	// 点群データ
	std::vector<Vec3> point_cloud(lidar.get_points_num());

	// icp_resourceの作成
	auto icp = icp_resource::to_resource<icp_resource::NormalKnownNonSyclResource>(
		lidar.get_points_num(),
		std::span{objects}
	);

	// 現在操作されているオブジェクト
	u32 current_controlled_object = 0;
	// 操作用タイマー
	sim::LapTimer timer{};

	// 制御周期タイマー
	sim::LapTimer control_duration_timer{};

	// 時間計測
	sim::LapTimer bench_timer{};

	auto update = [&](const my_pango_util::KeyboardHandler& keys) {
		control_duration_timer.sleep_for(std::chrono::duration<float>(1.f / scan_hz));
		bench_timer.clear();
		// 入力を処理
		{
			for (u8 iobj = 0; iobj < std::min<u64>(10, objects.size()); ++iobj) {
				if (keys['0' + iobj]) { current_controlled_object = iobj; }
			}

			Vec3 p{};
			p.x() = keys['a'] ? -1.f : keys['d'] ? 1.f : 0.f;
			p.y() = keys['e'] ? -1.f : keys['z'] ? 1.f : 0.f;
			p.z() = keys['w'] ? -1.f : keys['x'] ? 1.f : 0.f;

			const float roll = keys['h'] ? -1.f : keys['k'] ? 1.f : 0.f;
			const float pitch = keys['i'] ? -1.f : keys['n'] ? 1.f : 0.f;
			const float yaw = keys['u'] ? -1.f : keys['m'] ? 1.f : 0.f;
			// const float yaw = 1.f;

			const float dt = timer.lap().count();
			const auto diff = SE3::trans(trans_speed * dt * p)
				* SE3::rot(quaternion::ypr(rot_speed * dt * Vec3{roll, pitch, yaw}));

			true_poses[current_controlled_object] = diff * true_poses[current_controlled_object];
			std::println("pose_t 0: {}", Repr<SE3>::repr(true_poses[0]));
		}
		std::println("process_input: {}", bench_timer.lap());

		// 点群を生成
		{
			for (u32 ip = 0; ip < lidar.get_points_num(); ++ip) {
				const auto [true_ray, noised_ray] = lidar.generate_ray(ip, t, rand_gen);

				float dist = std::numeric_limits<float>::infinity();
				for (u8 iobj = 0; iobj < objects.size(); ++iobj) {
					for (const auto& surf : objects[iobj]) {
						std::visit(
							[&](const auto& surf) noexcept {
								auto moved_surf = surf;
								moved_surf.apply_se3(true_poses[iobj]);
								const auto res = moved_surf.ray_collision(true_ray);
								if (res < dist) { dist = res; }
							},
							surf
						);
					}
				}

				point_cloud[ip] =
					lidar.add_distance_noise(math::sqrt(dist), 0.8, rand_gen) * Vec3{noised_ray};
				// point_cloud[ip] = Vec{true_ray} * 2.f;
			}
			t += lidar.get_scan_time();
		}
		std::println("generate_points: {}", bench_timer.lap());

		// ICP
		{
			icp.run_icp(std::vector{point_cloud}, tikhnov, loop_num, pow2(accept_distance));
			for (u8 iobj = 0; iobj < objects.size(); ++iobj) {
				estimated_poses[iobj] = icp.obj_poses[iobj];
			}
			std::println("pose_e 0: {}", Repr<SE3>::repr(estimated_poses[0]));
		}
		std::println("icp: {}", bench_timer.lap());
	};

	auto draw = [&](const my_pango_util::KeyboardHandler&) {
		bench_timer.clear();
		// 描画
		// my_pango_util::draw::drawPose(SE3::ide(), {255.f, 0.f, 0.f}, {0.f, 255.f, 0.f}, {0.f, 0.f, 255.f});
		for (const auto& true_pose : true_poses) {
			my_pango_util::draw::drawPose(
				true_pose,
				0.5,
				{255.f, 0.f, 0.f},
				{0.f, 255.f, 0.f},
				{0.f, 0.f, 255.f}
			);
		}
		for (const auto& estimated_pose : estimated_poses) {
			my_pango_util::draw::drawPose(
				estimated_pose,
				0.8,
				{128.f, 0.f, 0.f},
				{0.f, 128.f, 0.f},
				{0.f, 0.f, 128.f}
			);
		}
		my_pango_util::draw::drawPointCloud(point_cloud, {255.f, 255.f, 255.f});
		std::println("draw: {}", bench_timer.lap());
	};

	my_pango_util::Pango pang{std::move(update), std::move(draw)};

	pang.run();
}