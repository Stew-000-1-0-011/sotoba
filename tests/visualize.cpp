#include <chrono> // 時間を扱うために必要
#include <mutex> // Mutexを使うために必要
#include <random>
#include <stop_token>
#include <syncstream>
#include <thread> // スレッドを使うために必要
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pangolin/pangolin.h>

#include <doctest.h>

#include "sotoba/cylinder.hpp"
#include "sotoba/fundamental.hpp"
#include "sotoba/lib.hpp"
#include "sotoba/lidar.hpp"
#include "sotoba/rectangle.hpp"

using namespace sotoba;

// 描画するデータの型を定義
using PointCloud = std::vector<Eigen::Vector3f>;
using Pose = Eigen::Isometry3f; // 4x4の変換行列で姿勢を表現

auto to_pose(const sotoba::SE3& se3) -> Pose {
	// 1. sotobaのクォータニオン(x,y,z,w)をEigenのクォータニオンに変換
	// 注意: Eigen::Quaternionfのコンストラクタは(w, x, y, z)の順序
	const auto& q_sycl = se3.q.v;
	Eigen::Quaternionf q_eigen(q_sycl.w(), q_sycl.x(), q_sycl.y(), q_sycl.z());

	// 2. sotobaの並進ベクトル(x,y,z)をEigenのベクトルに変換
	const auto& p_sycl = se3.p;
	Eigen::Vector3f p_eigen(p_sycl.x(), p_sycl.y(), p_sycl.z());

	// 3. EigenのクォータニオンとベクトルからIsometry3fを構築
	Eigen::Isometry3f isometry = Eigen::Isometry3f::Identity();
	isometry.rotate(q_eigen);
	isometry.pretranslate(p_eigen);

	return isometry;
}

// --- 描画関数 ---
void drawOrigin() {
	glPointSize(30.f);
	glBegin(GL_POINTS);
	glColor3f(1.f, 0.f, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glEnd();
}

// 点群を描画する
void drawPointCloud(const PointCloud& points, const Vec3& rgb) {
	glPointSize(3.0f);
	glBegin(GL_POINTS);
	glColor3f(rgb.x(), rgb.y(), rgb.z()); // 点の色
	for (const auto& p : points) { glVertex3f(p.x(), p.y(), p.z()); }
	glEnd();
}

// 姿勢（座標フレーム）を描画する
void drawPose(const Pose& pose, const Vec3& xrgb, const Vec3& yrgb, const Vec3& zrgb) {
	;
	glPushMatrix();
	glMultMatrixf(pose.matrix().data());
	glLineWidth(3);

	// 線の描画を開始
	glBegin(GL_LINES);

	// X軸
	glColor3f(xrgb.x(), xrgb.y(), xrgb.z());
	glVertex3f(0.0f, 0.0f, 0.0f); // 始点
	glVertex3f(0.5f, 0.0f, 0.0f); // 終点 (長さ0.5)

	// Y軸
	glColor3f(yrgb.x(), yrgb.y(), yrgb.z());
	glVertex3f(0.0f, 0.0f, 0.0f); // 始点
	glVertex3f(0.0f, 0.5f, 0.0f); // 終点

	// Z軸
	glColor3f(zrgb.x(), zrgb.y(), zrgb.z());
	glVertex3f(0.0f, 0.0f, 0.0f); // 始点
	glVertex3f(0.0f, 0.0f, 0.5f); // 終点

	glEnd(); // 線の描画を終了

	glPopMatrix();
}

// 1. カスタムハンドラクラスを定義
struct VelocityController: public pangolin::Handler3D {
	SE3 velocity_;
	std::mutex mtx_{};

	virtual ~VelocityController() = default;

	// VelocityControllerのコンストラクタ
	VelocityController(pangolin::OpenGlRenderState& cam_state)
		: pangolin::Handler3D(cam_state), velocity_(SE3::ide()) {}

	// キーが押された/離された時にPangolinによって呼ばれる
	virtual void
	Keyboard(pangolin::View&, unsigned char key, int /*x*/, int /*y*/, bool pressed) override {
		constexpr float trans_speed = 0.2f;
		constexpr float angular_speed = 0.25f;

		std::lock_guard lck{mtx_};
		// 移動速度の更新
		switch (key) {
			case 'w':
				velocity_.p.x() = pressed ? trans_speed : 0.0f;
				break;
			case 's':
				velocity_.p.x() = pressed ? -trans_speed : 0.0f;
				break;
			case 'a':
				velocity_.p.y() = pressed ? trans_speed : 0.0f;
				break;
			case 'd':
				velocity_.p.y() = pressed ? -trans_speed : 0.0f;
				break;
			case 'q':
				velocity_.p.z() = pressed ? trans_speed : 0.0f;
				break;
			case 'e':
				velocity_.p.z() = pressed ? -trans_speed : 0.0f;
				break;
		}

		// 回転速度の更新
		switch (key) {
			case 'j':
				velocity_.q =
					UnitQuaternion::from_rpy(Vec3{0.f, 0.f, pressed ? angular_speed : 0.0f});
				break; // Yaw
			case 'l':
				velocity_.q =
					UnitQuaternion::from_rpy(Vec3{0.f, 0.f, pressed ? -angular_speed : 0.0f});
				break;
			case 'i':
				velocity_.q =
					UnitQuaternion::from_rpy(Vec3{0.f, pressed ? angular_speed : 0.0f, 0.f});
				break; // Pitch
			case 'k':
				velocity_.q =
					UnitQuaternion::from_rpy(Vec3{0.f, pressed ? -angular_speed : 0.0f, 0.f});
				break;
			case 'u':
				velocity_.q =
					UnitQuaternion::from_rpy(Vec3{pressed ? angular_speed : 0.0f, 0.f, 0.f});
				break; // Roll
			case 'o':
				velocity_.q =
					UnitQuaternion::from_rpy(Vec3{pressed ? -angular_speed : 0.0f, 0.f, 0.f});
				break;
		}
	}

	// データ更新スレッドからこの関数を呼び出す
	void ApplyTo(SE3& pose) {
		SE3 vel;
		{
			std::lock_guard lck{mtx_};
			vel = velocity_;
		}

		pose = pose * vel;
		pose.normalize();
	}
};

TEST_SUITE("Visualize") {
	TEST_CASE("main") {
		// --- 1. Pangolinの初期化 ---
		pangolin::CreateWindowAndBind("Pangolin Real-time Sample", 1024, 768);
		glEnable(GL_DEPTH_TEST);

		// 3Dビューとカメラの設定
		pangolin::OpenGlRenderState s_cam(
			pangolin::ProjectionMatrix(1024, 768, 420, 420, 512, 384, 0.1, 1000),
			pangolin::ModelViewLookAt(-2, -2, -3, 0, 0, 0, pangolin::AxisY)
		);

		auto velocity_controller = VelocityController(s_cam);
		pangolin::View& d_cam = pangolin::CreateDisplay()
									.SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
									.SetHandler(&velocity_controller);

		// 描画するデータを格納する変数
		PointCloud true_point_cloud;
		PointCloud observed_point_cloud;
		Pose true_pose;
		Pose estimated_pose;
		std::mutex data_mutex; // データアクセスを保護するためのMutex
		std::stop_source ssource{};

		const Lidar3d mid360 = {
			// .vertcal_min = -std::numbers::pi * -7.f / 180.f,
			// .vertcal_max = std::numbers::pi * 52.f / 180.f,
			.vertcal_min = -std::numbers::pi,
			.vertcal_max = std::numbers::pi,
			.horizontal_min = 0.f,
			.horizontal_max = static_cast<float>(2.0 * std::numbers::pi),
			.range_precision = 0.02f,
			.angular_precision = 0.15f / 180.f * std::numbers::pi_v<float>,
			.vertical_num = 500,
			.horizontal_num = 40,
		};

		true_point_cloud.resize(mid360.num());
		observed_point_cloud.resize(mid360.num());

		// --- 2. データ更新スレッドを開始 ---
		std::thread data_thread([&,
									stoken = ssource.get_token(),
									velocity_controller_p = &velocity_controller] {
			try {
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
				};
				std::vector<CylinderOuter> global_cyls{
					{.center_and_radius = {0.f, 0.f, 0.f, 0.f},
						.axis_and_hheight = {0.f, 0.f, 1.f, 0.f}}
				};

				SE3 true_pose_{UnitQuaternion::one(), Vec4{}};
				SE3 estimated_pose_ = true_pose_
					* SE3{UnitQuaternion::from_rpy({0.1f, 0.1f, 0.1f}), {0.5f, 0.5f, 0.5f, 0.0f}};
				estimated_pose_.normalize();
				auto unexpended_time = std::chrono::duration<double>::zero();

				std::default_random_engine eng{};
				std::uniform_real_distribution<float> dist{-0.3f, 0.3f};
				std::uniform_real_distribution<float> ang{-0.2f, 0.2f};

				std::vector<Vec4> true_points(mid360.num());
				std::vector<Vec4> observed_points(mid360.num());

				u32 seed = 0;
				u32 total_step = 0;
				u32 fail_count = 0;
				sycl::queue q{sycl::cpu_selector_v};
				sycl::buffer<Rectangle> global_rects_buf(global_rects);
				sycl::buffer<Rectangle> local_rects_buf(sycl::range(global_rects.size()));
				sycl::buffer<CylinderOuter> global_cyls_buf(global_cyls);
				sycl::buffer<CylinderOuter> local_cyls_buf(sycl::range(global_cyls.size()));
				sycl::buffer<Vec4> lidar_true_points_buf(true_points.data(), mid360.num());
				sycl::buffer<Vec4> lidar_observed_points_buf(observed_points.data(), mid360.num());
				sycl::buffer<Vec4> closest_points_buf(sycl::range(mid360.num()));
				sycl::buffer<Ray> global_rays_buf(sycl::range(mid360.num()));

				while (!stoken.stop_requested()) {
					// 10Hz (100ms) ごとにデータを更新
					std::this_thread::sleep_for(std::chrono::milliseconds(100));

					// true_pose_の移動
					velocity_controller_p->ApplyTo(true_pose_);

					// 時間をとめてシミュレーション
					const auto unexpended_start = std::chrono::steady_clock::now();
					sotoba::simulate_lidar(
						q,
						mid360,
						true_pose_,
						seed++,
						global_rects_buf,
						global_cyls_buf,
						global_rays_buf,
						lidar_true_points_buf,
						lidar_observed_points_buf
					);
					const auto unexpeded_end = std::chrono::steady_clock::now();
					unexpended_time += unexpeded_end - unexpended_start;
					// std::osyncstream{
					// 	std::cerr
					// } << "eraplsed: "
					//   << std::chrono::duration_cast<std::chrono::duration<double>>(unexpended_time)
					// 		 .count()
					//   << std::endl;

					// 自己位置推定
					const auto new_estimated_pose_ = sotoba::icp_self_localization(
						q,
						estimated_pose_,
						10,
						lidar_observed_points_buf,
						closest_points_buf,
						global_rects_buf,
						global_cyls_buf,
						local_rects_buf,
						local_cyls_buf
					);

					const SE3 diff_pose = new_estimated_pose_ * !estimated_pose_;
					++total_step;
					if (test::Check{diff_pose.q} == test::Check{UnitQuaternion::one(), 0.1f}
						&& test::Check{diff_pose.p} == test::Check{Vec4{}, 0.6f}) {
						estimated_pose_ = new_estimated_pose_;
					}
					else {
						++fail_count;
						std::osyncstream{std::cerr} << "Fail!: " << double(fail_count) / total_step << std::endl;
					}

					// Mutexを使ってグローバル変数を安全に更新
					{
						std::lock_guard<std::mutex> lock(data_mutex);
						auto true_points_acc = lidar_true_points_buf.get_host_access();
						auto observed_points_acc = lidar_observed_points_buf.get_host_access();

						for (u32 i = 0; i < mid360.num(); ++i)
							true_point_cloud[i] = {
								true_points_acc[i].x(),
								true_points_acc[i].y(),
								true_points_acc[i].z()
							};
						for (u32 i = 0; i < mid360.num(); ++i)
							observed_point_cloud[i] = {
								observed_points_acc[i].x(),
								observed_points_acc[i].y(),
								observed_points_acc[i].z()
							};
						true_pose = to_pose(true_pose_);
						estimated_pose = to_pose(estimated_pose_);
					}
					// std::osyncstream{std::cerr} << "diff: " << test::Check{true_pose_} << ' '
					// 							<< test::Check{estimated_pose_} << std::endl;
				}
			} catch (const std::exception& e) {
				std::osyncstream{std::cerr} << e.what() << std::endl;
			}
		});

		PointCloud current_true_points(mid360.num());
		PointCloud current_observed_points(mid360.num());
		Pose current_true_pose{};
		Pose current_estimated_pose{};
		// --- 3. メインの描画ループ ---
		while (!pangolin::ShouldQuit()) {
			// 画面をクリア
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // 背景色 (濃いグレー)
			d_cam.Activate(s_cam);

			// ローカル変数に描画データをコピー

			{
				std::lock_guard<std::mutex> lock(data_mutex);
				for (u32 i = 0; i < mid360.num(); ++i) current_true_points[i] = true_point_cloud[i];
				for (u32 i = 0; i < mid360.num(); ++i)
					current_observed_points[i] = observed_point_cloud[i];
				current_true_pose = true_pose;
				current_estimated_pose = estimated_pose;
			}

			// 描画関数を呼び出す
			drawOrigin();
			drawPointCloud(current_true_points, {1.f, 1.f, 1.f});
			drawPointCloud(current_observed_points, {1.f, 1.f, 0.f});
			drawPose(current_true_pose, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f});
			drawPose(current_estimated_pose, {0.f, 1.f, 1.f}, {1.f, 0.f, 1.f}, {1.f, 1.f, 0.f});

			// 画面を更新
			pangolin::FinishFrame();
		}
		ssource.request_stop();
		data_thread.join(); // スレッドの終了を待つ
	}
}