#pragma once

#include <atomic>
#include <concepts>
#include <type_traits>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <pangolin/pangolin.h>

#include "sotoba/math/quaternion.hpp"
#include "sotoba/math/se3.hpp"
#include "sotoba/math/square_mat.hpp"
#include "sotoba/math/vec.hpp"

namespace my_pango_util::impl {
	using namespace sotoba;
	using namespace math;
	using Eigen::Matrix4f;

	inline auto pose_to_mat4f(const SE3& h) -> Matrix4f {
		const auto rot = quaternion::to_mat(h.uq);
		const auto t = h.p;

		Matrix4f ret = Matrix4f::Identity();
		for(u8 i = 0; i < 3; ++i) for(u8 j = 0; j < 3; ++j) {
			ret(i, j) = rot[i, j];
		}
		for(u8 i = 0; i < 3; ++i) ret(3, i) = t[i];
		ret(3, 3) = 1;

		return ret;
	}

	namespace draw {
		// 原点を描画する
		inline void drawOrigin() {
			glPointSize(10.f);
			glBegin(GL_POINTS);
			glColor3f(1.f, 0.f, 0.f);
			glVertex3f(0.f, 0.f, 0.f);
			glEnd();
		}

		// 点群を描画する
		inline void drawPointCloud(const std::span<Vec3>& points, const Vec3& rgb) {
			glPointSize(3.0f);
			glBegin(GL_POINTS);
			glColor3f(rgb.x(), rgb.y(), rgb.z()); // 点の色
			for (const auto& p : points) { glVertex3f(p.x(), p.y(), p.z()); }
			glEnd();
		}

		// 姿勢（座標フレーム）を描画する
		inline void drawPose(const SE3& pose, const float len, const Vec3& xrgb, const Vec3& yrgb, const Vec3& zrgb) {
			glPushMatrix();
			const auto mat = pose_to_mat4f(pose);
			glMultMatrixf(mat.data());
			glLineWidth(3);

			// 線の描画を開始
			glBegin(GL_LINES);

			// X軸
			glColor3f(xrgb.x(), xrgb.y(), xrgb.z());
			glVertex3f(0.0f, 0.0f, 0.0f); // 始点
			glVertex3f(len, 0.0f, 0.0f); // 終点 (長さ0.5)

			// Y軸
			glColor3f(yrgb.x(), yrgb.y(), yrgb.z());
			glVertex3f(0.0f, 0.0f, 0.0f); // 始点
			glVertex3f(0.0f, len, 0.0f); // 終点

			// Z軸
			glColor3f(zrgb.x(), zrgb.y(), zrgb.z());
			glVertex3f(0.0f, 0.0f, 0.0f); // 始点
			glVertex3f(0.0f, 0.0f, len); // 終点

			glEnd(); // 線の描画を終了

			glPopMatrix();
		}
	}

	// 1. カスタムハンドラクラスを定義
	struct KeyboardHandler: public pangolin::Handler3D {
		std::vector<std::atomic<bool>> is_pushed;

		virtual ~KeyboardHandler() = default;

		// KeyboardHandlerのコンストラクタ
		KeyboardHandler(pangolin::OpenGlRenderState& cam_state)
			: pangolin::Handler3D(cam_state), is_pushed(127) {}

		// キーが押された/離された時にPangolinによって呼ばれる
		virtual void
		Keyboard(pangolin::View&, unsigned char key, int /*x*/, int /*y*/, bool pressed) override {
			this->is_pushed[key].store(pressed, std::memory_order::relaxed);
		}

		auto operator[](unsigned char key) const noexcept -> bool {
			return this->is_pushed[key].load(std::memory_order::relaxed);
		}
	};

	template <class UpdateF_, class DrawF_>
	struct Pango final {
		int dummy;
		pangolin::OpenGlRenderState s_cam;
		KeyboardHandler keyboard;
		pangolin::View& d_cam;
		UpdateF_ update_f;
		DrawF_ draw_f;

		Pango(std::invocable<const KeyboardHandler&> auto&& update_f, std::invocable<const KeyboardHandler&> auto&& draw_f)
		: dummy{[] {
			pangolin::CreateWindowAndBind("sotoba simulation", 1024, 768);
			glEnable(GL_DEPTH_TEST);
			return 0;
		}()}
		, s_cam{pangolin::ProjectionMatrix(1024, 768, 420, 420, 512, 384, 0.1, 1000), pangolin::ModelViewLookAt(-2, -2, -3, 0, 0, 0, pangolin::AxisY)}
		, keyboard{s_cam}
		, d_cam{pangolin::CreateDisplay()
					.SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
					.SetHandler(&this->keyboard)}
		, update_f{std::forward<decltype(update_f)>(update_f)}
		, draw_f{std::forward<decltype(draw_f)>(draw_f)}
		{}

		void run() {
			while (!pangolin::ShouldQuit()) {
				this->update_f(this->keyboard);

				// 画面をクリア
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // 背景色 (濃いグレー)
				this->d_cam.Activate(this->s_cam);

				this->draw_f(this->keyboard);

				// 画面を更新
				pangolin::FinishFrame();
			}
		}
	};
	Pango(std::invocable<const KeyboardHandler&> auto&& update_f, std::invocable<const KeyboardHandler&> auto&& draw_f) -> Pango<std::remove_cvref_t<decltype(update_f)>, std::remove_cvref_t<decltype(draw_f)>>;
} // namespace my_pango_util::impl

namespace my_pango_util {
	using impl::KeyboardHandler;
	using impl::Pango;
	namespace draw = impl::draw;
}