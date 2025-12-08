#pragma once

#include "sotoba/math/approx_check.hpp"
#include "sotoba/math/epsilon.hpp"
#include "sotoba/math/se3.hpp"
#include "sotoba/math/vec.hpp"
#include "sotoba/surface/surface.hpp"

namespace sotoba::surface::rectangle_impl {
	using sotoba::math::epsilon;
	using sotoba::math::SE3;
	using sotoba::math::Vec;
	using sotoba::math::Vec3;
	using sotoba::math::UVec3;
	using sotoba::math::Vec4;
	namespace math = sotoba::math;
	namespace vec = sotoba::math::vec;
	

	struct Rectangle final {
		Vec3 center;
		float dummy1;
		Vec4 u_axis_and_hlen; // .xyz = u_axis (単位ベクトル), .w = half_u_len
		Vec4 v_axis_and_hlen; // .xyz = v_axis (単位ベクトル), .w = half_v_len
		UVec3 normal;
		float dummy2;

		Rectangle(const Vec3& center, const Vec4& u_axis_and_hlen, const Vec4& v_axis_and_hlen, const UVec3& normal) noexcept
		: center{center}
		, dummy1{0.f}
		, u_axis_and_hlen{u_axis_and_hlen}
		, v_axis_and_hlen{v_axis_and_hlen}
		, normal{normal}
		, dummy2{0.f}
		{}

		Rectangle(const Rectangle&) noexcept = default;
		Rectangle(Rectangle&&) noexcept = default;
		auto operator=(const Rectangle&) -> Rectangle& = default;
		auto operator=(Rectangle&&) -> Rectangle& = default;

		auto closest_pd(const Vec3& p) const noexcept -> Vec4 {
			const auto cp = p - this->center;
			// Project d onto the rectangle's local axes
			float u_dist = vec::dot(cp, this->u_axis_and_hlen.xyz());
			float v_dist = vec::dot(cp, this->v_axis_and_hlen.xyz());
			// Clamp the distances to the rectangle's extents
			u_dist = math::clamp(u_dist, -this->u_axis_and_hlen.w(), this->u_axis_and_hlen.w());
			v_dist = math::clamp(v_dist, -this->v_axis_and_hlen.w(), this->v_axis_and_hlen.w());
			// The closest point is the center plus the clamped projections
			const auto q = this->center + u_dist * this->u_axis_and_hlen.xyz()
				+ v_dist * this->v_axis_and_hlen.xyz();
			const auto dot = vec::dot(-q, Vec3{this->normal});
			// qの原点からの可視性チェック
			// 見えるはずのない点には距離無限で返す
			if (dot < 0.0f) {
				return {Vec3{}, Vec{std::numeric_limits<float>::infinity()}};
			}
			const auto qp = p - q;
			return {q, Vec{vec::dot(qp, qp)}};
		}

		auto closest_pdn(const Vec3& p) const noexcept -> std::pair<Vec4, UVec3> {
			const auto cp = p - this->center;
			// Project d onto the rectangle's local axes
			float u_dist = vec::dot(cp, this->u_axis_and_hlen.xyz());
			float v_dist = vec::dot(cp, this->v_axis_and_hlen.xyz());
			// Clamp the distances to the rectangle's extents
			u_dist = math::clamp(u_dist, -this->u_axis_and_hlen.w(), this->u_axis_and_hlen.w());
			v_dist = math::clamp(v_dist, -this->v_axis_and_hlen.w(), this->v_axis_and_hlen.w());
			// The closest point is the center plus the clamped projections
			const auto q = this->center + u_dist * this->u_axis_and_hlen.xyz()
				+ v_dist * this->v_axis_and_hlen.xyz();
			const auto dot = vec::dot(-q, Vec3{this->normal});
			// qの原点からの可視性チェック
			// 見えるはずのない点には距離無限で返す
			if (dot < 0.0f) {
				return {Vec4{Vec3{}, Vec{std::numeric_limits<float>::infinity()}}, {}};
			}
			const auto qp = p - q;
			return {{q, Vec{vec::dot(qp, qp)}}, this->normal};
		}

		void apply_se3(const SE3& h) noexcept {
			this->center = h.app_v(this->center);
			this->u_axis_and_hlen = {
				h.app_uv(this->u_axis_and_hlen.xyz()),
				Vec{this->u_axis_and_hlen.w()}
			};
			this->v_axis_and_hlen = {
				h.app_uv(this->v_axis_and_hlen.xyz()),
				Vec{this->v_axis_and_hlen.w()}
			};
			this->normal = h.app_uv(this->normal);
		}

		auto ray_collision(const UVec3& ray) const noexcept -> float {
			const float denom = vec::dot(this->normal, ray);

			// レイが平面と平行な場合 (内積がほぼ0)
			if (math::fabs(denom) < epsilon) {
				return std::numeric_limits<float>::infinity();
			}

			const auto origin_to_p = this->center;
			const float t = vec::dot(origin_to_p, this->normal) / denom;

			// 交点がレイの始点より後ろにある場合
			if (t < epsilon) return std::numeric_limits<float>::infinity();

			const auto intersection_point = t * ray;
			const auto vec_from_center = intersection_point - this->center;

			// 交点が長方形の範囲内にあるかチェック
			const float u_dist = vec::dot(vec_from_center, Vec3{this->u_axis_and_hlen.xyz()});
			const float v_dist = vec::dot(vec_from_center, Vec3{this->v_axis_and_hlen.xyz()});

			if (math::fabs(u_dist) <= this->u_axis_and_hlen.w()
				&& math::fabs(v_dist) <= this->v_axis_and_hlen.w()) {
				const float dist_sq = vec::dot(intersection_point, intersection_point);
				return dist_sq;
			}

			// 長方形の範囲外
			return std::numeric_limits<float>::infinity();
		}
	};
	static_assert(surfacelike<Rectangle>);
}

namespace sotoba::surface {
	using rectangle_impl::Rectangle;
}

#ifdef sotoba_ENABLE_TESTING
#include <numbers>

#include <doctest.h>

#include "sotoba/math/approx_check.hpp"
#include "sotoba/math/quaternion.hpp"

TEST_SUITE("rectangle.hpp") {
	using namespace sotoba;
	using surface::Rectangle;
	using math::Vec;
	using math::Vec3;
	using math::Vec4;
	using math::UVec3;
	using math::SE3;
	namespace vec = math::vec;
	namespace quaternion = math::quaternion;
	using math::ApproxCheck;

	using std::numbers::pi;

	// 基本的な矩形のセットアップ
	// 中心: (0, 0, 5)
	// 法線: (0, 0, -1) -> 原点(0,0,0)を向いている (可視性チェックを通すため)
	// U軸: X軸 (幅 4 -> half 2.0)
	// V軸: Y軸 (高さ 2 -> half 1.0)
	inline Rectangle rect {
		Vec3{0.0f, 0.0f, 5.0f}, // center
		Vec4{1.0f, 0.0f, 0.0f, 2.0f}, // u_axis (X), hlen = 2.0
		Vec4{0.0f, 1.0f, 0.0f, 1.0f}, // v_axis (Y), hlen = 1.0
		UVec3{0.0f, 0.0f, -1.0f},     // normal (facing -Z)
	};

	TEST_CASE("closest_pd: Closest Point and Distance") {
		SUBCASE("Point directly 'above' the center (inside rectangle)") {
			// 矩形の中心(0,0,5)の手前 (0,0,0) からの距離
			Vec3 p{0.0f, 0.0f, 0.0f};
			auto res = rect.closest_pd(p);

			// 最近接点 q は center と同じはず
			CHECK(ApproxCheck{Vec3{res.xyz()}} == ApproxCheck{Vec3{0.f, 0.f, 5.f}});

			// 距離の二乗: p(0,0,0) と q(0,0,5) の距離^2 = 25
			CHECK(res.w() == doctest::Approx(25.0f));
		}

		SUBCASE("Point projected falls within bounds") {
			// (1.5, 0.5, 0) -> 投影すると (1.5, 0.5, 5)
			// U範囲(-2~2), V範囲(-1~1) なのでクランプされない
			Vec3 p{1.5f, 0.5f, 0.0f};
			auto res = rect.closest_pd(p);

			CHECK(ApproxCheck{Vec3{res.xyz()}} == ApproxCheck{Vec3{1.5f, 0.5f, 5.f}});
			
			// p(1.5, 0.5, 0) - q(1.5, 0.5, 5) = (0, 0, -5) -> dist_sq = 25
			CHECK(res.w() == doctest::Approx(25.0f));
		}

		SUBCASE("Point outside bounds (Clamping behavior)") {
			// U軸方向に範囲外 (3.0, 0, 0) -> half_len 2.0 にクランプされるはず
			Vec3 p{3.0f, 0.0f, 0.0f};
			auto res = rect.closest_pd(p);

			// 最近接点は (2.0, 0.0, 5.0)
			CHECK(res.x() == doctest::Approx(2.0f));
			CHECK(res.y() == doctest::Approx(0.0f));
			CHECK(res.z() == doctest::Approx(5.0f));

			// 距離計算: p(3,0,0) - q(2,0,5) = (1, 0, -5) -> 1 + 25 = 26
			CHECK(res.w() == doctest::Approx(26.0f));
		}

		SUBCASE("Visibility Check (Backface Culling logic)") {
			// 矩形は -Z (原点方向) を向いている。
			// もし矩形が +Z を向いていたら、原点からは「裏側」が見えていることになり、無限大を返すはず。
			
			Rectangle back_facing_rect = rect;
			back_facing_rect.normal = UVec3{0.0f, 0.0f, 1.0f}; // 原点とは逆向き

			Vec3 p{0.0f, 0.0f, 0.0f};
			auto res = back_facing_rect.closest_pd(p);

			// dot(-q, normal) < 0 のチェック
			// q=(0,0,5), -q=(0,0,-5), normal=(0,0,1) -> dot = -5 < 0
			// 結果は無限大のはず
			CHECK(res.w() == std::numeric_limits<float>::infinity());
		}
	}

	TEST_CASE("ray_collision: Intersection Test (Ray from Origin)") {
	// テスト用セットアップ
	// 中心: (0, 0, 5)
	// 法線: (0, 0, -1) -> 原点の方を向いている
	// サイズ: 幅4 (half=2), 高さ2 (half=1)

	SUBCASE("Direct Hit (Center)") {
		// 原点から (0,0,5) に向かう方向 -> Z軸プラス方向 (0,0,1)
		UVec3 ray_dir{0.0f, 0.0f, 1.0f};

		auto res = rect.ray_collision(ray_dir);
		
		// 期待値: 衝突点は (0, 0, 5)		
		// 距離の二乗: 原点(0,0,0)と(0,0,5)の距離^2 = 25
		CHECK(res == doctest::Approx(25.0f));
	}

	SUBCASE("Hit (Inside Bounds)") {
		// 矩形上の点 (1.5, 0.5, 5.0) を狙う
		// これは U範囲(±2.0), V範囲(±1.0) の内側
		Vec3 target{1.5f, 0.5f, 5.0f};
		
		// 方向ベクトルを作成 (正規化が必要と想定されるため正規化する)
		UVec3 ray_dir{vec::fast_normalize(target)};
		auto res = rect.ray_collision(ray_dir);

		// 衝突点は target そのものになるはず
		CHECK(ApproxCheck{Vec{ray_dir} * math::sqrt(res)} == ApproxCheck{Vec3{1.5f, 0.5f, 5.f}});
	}

	SUBCASE("Miss (Outside Bounds)") {
		// 平面 Z=5 上の点 (3.0, 0.0, 5.0) を狙う
		// X=3.0 は U_hlen=2.0 の外側なのでミスになるはず
		Vec3 target{3.0f, 0.0f, 5.0f};
		UVec3 ray_dir{vec::fast_normalize(target)}; 

		auto res = rect.ray_collision(ray_dir);

		// 無限大が返る
		CHECK(res == std::numeric_limits<float>::infinity());
	}

	SUBCASE("Miss (Backwards)") {
		// 原点から背中側 (Zマイナス方向)
		UVec3 ray_dir{0.0f, 0.0f, -1.0f};

		auto res = rect.ray_collision(ray_dir);

		CHECK(res == std::numeric_limits<float>::infinity());
	}
	
	SUBCASE("Miss (Parallel)") {
		// 平面 (法線 Z軸) に対して平行なレイ (X軸方向)
		UVec3 ray_dir{1.0f, 0.0f, 0.0f};

		auto res = rect.ray_collision(ray_dir);

		CHECK(res == std::numeric_limits<float>::infinity());
	}
}

	TEST_CASE("Rectangle::apply_se3") {
		// --- テスト用ヘルパー: クォータニオン作成 ---
		const auto rot_y_90 = quaternion::ypr({0.f, float(pi / 2.), 0.f});

		// --- 初期状態の矩形 ---
		// 中心: (0, 0, 5)
		// U軸: X軸 (幅4 -> half 2.0)
		// V軸: Y軸 (高さ2 -> half 1.0)
		// 法線: -Z方向 (0, 0, -1)
		Rectangle base_rect {
			Vec3{0.0f, 0.0f, 5.0f},
			Vec4{1.0f, 0.0f, 0.0f, 2.0f}, // u_axis=(1,0,0), hlen=2.0
			Vec4{0.0f, 1.0f, 0.0f, 1.0f}, // v_axis=(0,1,0), hlen=1.0
			UVec3{0.0f, 0.0f, -1.0f},
		};

		SUBCASE("Translation only") {
			Rectangle rect = base_rect;
			
			// (10, -5, 0) だけ平行移動
			Vec3 translation{10.0f, -5.0f, 0.0f};
			SE3 trans_se3 = SE3::trans(translation);

			rect.apply_se3(trans_se3);

			// Center: (0,0,5) + (10,-5,0) -> (10, -5, 5)
			CHECK(ApproxCheck{rect.center} == ApproxCheck{Vec3{10.0f, -5.0f, 5.0f}});

			// 方向ベクトルとサイズは変化しないはず
			CHECK(ApproxCheck{Vec3{rect.u_axis_and_hlen.xyz()}} == ApproxCheck{Vec3{1.0f, 0.0f, 0.0f}}); // U = X
			CHECK(ApproxCheck{Vec3{rect.v_axis_and_hlen.xyz()}} == ApproxCheck{Vec3{0.0f, 1.0f, 0.0f}}); // V = Y
			CHECK(ApproxCheck{UVec3{rect.normal}} == ApproxCheck{UVec3{0.0f, 0.0f, -1.0f}});      // Normal = -Z

			// ハーフサイズ(w)が変わっていないこと
			CHECK(rect.u_axis_and_hlen.w() == doctest::Approx(2.0f));
			CHECK(rect.v_axis_and_hlen.w() == doctest::Approx(1.0f));
		}

		SUBCASE("Rotation only (90 degrees around Y-axis)") {
			Rectangle rect = base_rect;
			
			// Y軸周りに90度回転
			// 座標変換の期待値:
			// X軸 -> -Z軸
			// Y軸 -> Y軸 (不変)
			// Z軸 -> X軸
			SE3 rot_se3 = SE3::rot(rot_y_90);

			rect.apply_se3(rot_se3);

			// Center: 元の(0,0,5) はZ軸上にあるので、回転後はX軸上の(5,0,0)になるはず
			CHECK(ApproxCheck{rect.center} == ApproxCheck{Vec3{5.0f, 0.0f, 0.0f}});

			// U Axis: 元のX軸(1,0,0) -> 90度回転 -> -Z軸(0,0,-1)
			CHECK(ApproxCheck{Vec3{rect.u_axis_and_hlen.xyz()}} == ApproxCheck{Vec3{0.0f, 0.0f, -1.0f}});

			// V Axis: 元のY軸(0,1,0) -> 回転してもY軸のまま
			CHECK(ApproxCheck{Vec3{rect.v_axis_and_hlen.xyz()}} == ApproxCheck{Vec3{0.0f, 1.0f, 0.0f}});

			// Normal: 元の-Z軸(0,0,-1) -> 90度回転 -> -X軸(-1,0,0)
			CHECK(ApproxCheck{rect.normal} == ApproxCheck{UVec3{-1.0f, 0.0f, 0.0f}});

			// サイズ確認
			CHECK(rect.u_axis_and_hlen.w() == doctest::Approx(2.0f));
		}

		SUBCASE("Combined SE3 (Rotation + Translation)") {
			Rectangle rect = base_rect;

			// Y軸90度回転 してから (0, 10, 0) 平行移動
			// SE3 = Trans * Rot
			// app_v(v) = Rot(v) + Trans
			SE3 combined = SE3::trans(Vec3{0.0f, 10.0f, 0.0f}) * SE3::rot(rot_y_90);

			rect.apply_se3(combined);

			// Center計算:
			// 1. Rotate (0,0,5) -> (5,0,0)
			// 2. Translate (5,0,0) + (0,10,0) -> (5, 10, 0)
			CHECK(ApproxCheck{rect.center} == ApproxCheck{Vec3{5.0f, 10.0f, 0.0f}});

			// 方向ベクトルは回転のみ影響を受ける (平行移動は無視)
			// U Axis: X -> -Z
			CHECK(ApproxCheck{Vec3{rect.u_axis_and_hlen.xyz()}} == ApproxCheck{Vec3{0.0f, 0.0f, -1.0f}});
			// Normal: -Z -> -X
			CHECK(ApproxCheck{rect.normal} == ApproxCheck{UVec3{-1.0f, 0.0f, 0.0f}});
		}
		
		SUBCASE("Identity Transformation") {
			Rectangle rect = base_rect;
			rect.apply_se3(SE3::ide());
			
			CHECK(ApproxCheck{rect.center} == ApproxCheck{base_rect.center});
			CHECK(ApproxCheck{Vec3{rect.u_axis_and_hlen.xyz()}} == ApproxCheck{Vec3{base_rect.u_axis_and_hlen.xyz()}});
			CHECK(ApproxCheck{rect.normal} == ApproxCheck{base_rect.normal});
		}
	}
}

#endif
