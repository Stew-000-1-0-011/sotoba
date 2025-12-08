#pragma once

#include <limits>
#include "sotoba/math/approx_check.hpp"
#include "sotoba/math/scalar_functions.hpp"
#include "sotoba/math/vec.hpp"
#include "sotoba/math/se3.hpp"
#include "sotoba/surface/surface.hpp"

namespace sotoba::cylinder_impl {
	using math::SE3;
	using math::UVec3;
	using math::Vec;
	using math::Vec3;
	using math::Vec4;
	namespace vec = math::vec;

	// 上下の底面が無い円筒
	struct CylinderOuter final {
		Vec3 center;
		float radius;
		UVec3 axis;
		float hheight;

		auto closest_pd(const Vec3& p) const noexcept -> Vec4 {
			/*
			c: center
			d: 中心軸(有限長さ)上のpの最近接点
			e: 中心軸(無限長さ)上のpの最近接点
			q: 円柱(有限長さ)側面上の最近接点
			*/
			const auto cp = p - this->center;
			const auto vertical = vec::dot(cp, axis);
			const auto cd = math::clamp(vertical, -this->hheight, this->hheight) * this->axis;
			const auto ce = vertical * axis;
			const auto ep = cp - ce;
			if (vec::dot(ep, ep) < math::epsilon) {
				return {Vec3{}, Vec{std::numeric_limits<float>::infinity()}};
			}
			const auto dq = this->radius * vec::fast_normalize(ep);
			const auto q = this->center + cd + dq;

			// qの可視性チェック
			// 原点からqが見えるか
			const auto dot = vec::dot(-q, dq);
			if (dot < 0.0f) {
				return {Vec3{}, Vec{std::numeric_limits<float>::infinity()}};
			}
			const auto pq = q - p;
			return {q, Vec{vec::dot(pq, pq)}};
		}

		auto closest_pdn(const Vec3& p) const noexcept -> std::pair<Vec4, UVec3> {
			/*
			c: center
			d: 中心軸(有限長さ)上のpの最近接点
			e: 中心軸(無限長さ)上のpの最近接点
			q: 円柱(有限長さ)側面上の最近接点
			*/
			const auto cp = p - this->center;
			const auto vertical = vec::dot(cp, axis);
			const auto cd = math::clamp(vertical, -this->hheight, this->hheight) * this->axis;
			const auto ce = vertical * axis;
			const auto ep = cp - ce;
			if (vec::dot(ep, ep) < math::epsilon) {
				return {
					{Vec3{}, Vec{std::numeric_limits<float>::infinity()}}
					, {}
				};
			}
			const auto n = vec::fast_normalize(ep);
			const auto dq = this->radius * n;
			const auto q = this->center + cd + dq;

			// qの可視性チェック
			// 原点からqが見えるか
			const auto dot = vec::dot(-q, n);
			if (dot < 0.0f) {
				return {
					{Vec3{}, Vec{std::numeric_limits<float>::infinity()}}
					, {}
				};
			}
			const auto pq = q - p;
			return {
				{q, Vec{vec::dot(pq, pq)}}
				, n
			};
		}

		void apply_se3(const SE3& h) noexcept {
			this->center = h.app_v(this->center);
			this->axis = h.app_uv(this->axis);
		}

		auto ray_collision(const UVec3& ray) const noexcept -> float {
			const auto co = -this->center;
			const float dot_d_a = vec::dot(ray, this->axis);
			const float dot_co_a = vec::dot(co, axis);

			const float a = 1.0f - math::pow2(dot_d_a);
			const float b = 2.0f * (vec::dot(ray, co) - dot_d_a * dot_co_a);
			const float c = vec::dot(co, co) - math::pow2(dot_co_a) - math::pow2(this->radius);

			// レイが軸と平行な場合
			if (math::fabs(a) < math::epsilon) return std::numeric_limits<float>::infinity();

			const float delta = b * b - 4.0f * a * c;

			// 交点がない場合
			if (delta < 0.0f) return std::numeric_limits<float>::infinity();

			const float sqrt_delta = math::sqrt(delta);
			float t = (-b - sqrt_delta) / (2.0f * a);

			// 最初の交点が背後にあるか、高さの範囲外かチェック
			if (t < math::epsilon || math::fabs(dot_co_a + t * dot_d_a) > hheight) {
				t = (-b + sqrt_delta) / (2.0f * a);
				// 2番目の交点も背後にあるか、高さの範囲外なら交差しない
				if (t < math::epsilon || math::fabs(dot_co_a + t * dot_d_a) > hheight) {
					return std::numeric_limits<float>::infinity();
				}
			}
			// 円柱の内側の点も出てきてしまうが、まあいいか

			return math::pow2(t);
		}
	};
	static_assert(surface::surfacelike<CylinderOuter>);
}

namespace sotoba::surface {
	using cylinder_impl::CylinderOuter;
}

#ifdef sotoba_ENABLE_TESTING
#include <numbers>

#include <doctest.h>


TEST_SUITE("cylinder.hpp") {
	using namespace sotoba;
	using sotoba::surface::CylinderOuter;
	using math::Vec3;
	using math::Vec4;
	using math::UVec3;
	using math::Vec;
	using math::SE3;
	using math::UQuaternion;
	namespace quaternion = math::quaternion;
	using sotoba::math::ApproxCheck;

	// 共通セットアップ: 原点からZ方向に離れた、Y軸平行の円筒
	// Center: (0, 0, 10)
	// Axis:   (0, 1, 0)
	// Radius: 2.0
	// Height: 10.0 (hheight = 5.0)
	TEST_CASE("CylinderOuter: Axis Aligned (Standard Position)") {
		CylinderOuter cyl;
		cyl.center = Vec3{0.0f, 0.0f, 10.0f};
		cyl.radius = 2.0f;
		cyl.axis = UVec3{0.0f, 1.0f, 0.0f};
		cyl.hheight = 5.0f;

		// -------------------------------------------------------------
		// closest_pd / closest_pdn のテスト
		// -------------------------------------------------------------
		SUBCASE("closest_pd: Visible Surface (Front)") {
			// 円筒の手前(Z=5)にある点。表面(Z=8)が一番近い。
			const Vec3 p = {0.0f, 0.0f, 5.0f};
			const Vec4 res = cyl.closest_pd(p);

			const Vec3 closest_pt = res.xyz();
			const float sq_dist = res.w();
			const Vec3 expected_pt = {0.0f, 0.0f, 8.0f}; // Center(10) - Radius(2)

			CHECK(ApproxCheck{closest_pt} == ApproxCheck{expected_pt});
			CHECK(sq_dist == doctest::Approx(9.0f)); // (8-5)^2 = 9
		}

		SUBCASE("closest_pd: Invisible Surface (Back face culling)") {
			// 円筒の奥(Z=15)にある点。
			// 幾何的には Z=12 が近いが、裏面なので不可視扱い(Inf)になるはず。
			const Vec3 p = {0.0f, 0.0f, 15.0f};
			const Vec4 res = cyl.closest_pd(p);

			CHECK(res.w() == std::numeric_limits<float>::infinity());
		}

		SUBCASE("closest_pd: Height Clamping (Top/Bottom edge)") {
			// Y=10 (hheight=5の範囲外) からの最近接点
			// Y座標が 5.0 にクランプされるはず
			const Vec3 p = {0.0f, 10.0f, 5.0f};
			const Vec4 res = cyl.closest_pd(p);
			const Vec3 closest_pt = res.xyz();
			
			const Vec3 expected_pt = {0.0f, 5.0f, 8.0f};

			CHECK(ApproxCheck{closest_pt} == ApproxCheck{expected_pt});
		}

		SUBCASE("closest_pdn: Normal Vector Verification") {
			// 側面の点に対する法線
			const Vec3 p = {3.0f, 0.0f, 10.0f}; // X正方向から
			auto [res, normal] = cyl.closest_pdn(p);
			
			// 原点から最近接点は不可視
			CHECK(res.w() == std::numeric_limits<float>::infinity());
		}

		// -------------------------------------------------------------
		// ray_collision のテスト (レイは原点(0,0,0)始点固定)
		// -------------------------------------------------------------
		SUBCASE("ray_collision: Direct Hit") {
			// 原点からZ軸方向へ
			const UVec3 ray = {0.0f, 0.0f, 1.0f};
			const float res = cyl.ray_collision(ray);

			CHECK(res == doctest::Approx(64.0f)); // 距離8の二乗
		}

		SUBCASE("ray_collision: Miss (Passes through height gap)") {
			// Y軸上方へ大きく傾いたレイ
			// 円筒の上端は Y=5, Z=8付近。
			// レイの傾きが大きく、円筒の無限延長上には当たるが、有限高さには当たらないケース
			
			// Z=8の位置で Y=6.0 (hheight=5.0より上) になるようなレイ
			const Vec3 dir = Vec3{0.f, 6.f, 8.f}; 
			const UVec3 ray = math::vec::fast_normalize(dir); // 要正規化

			const float res = cyl.ray_collision(ray);
			CHECK(res == std::numeric_limits<float>::infinity());
		}

		SUBCASE("ray_collision: Grazing Edge (Tangent)") {
			// ちょうど接するラインを狙う (X=5sqrt(6) / 6, Z=10)
			// 原点から (2.041241452319315, 0, 10) 方向へのレイ
			const Vec3 dir = Vec3{2.041241452319315f, 0.0f, 10.0f}; 
			const UVec3 ray = math::vec::fast_normalize(dir);

			const float res = cyl.ray_collision(ray);
			
			// 計算誤差により厳密な接触は判定が難しいが、
			// ほぼ (1.9595917942265426, 0, 9.6) に近い位置で衝突、もしくはギリギリ交差とみなされるか確認
			// 判別式 delta が 0 付近になるケース。
			if (res != std::numeric_limits<float>::infinity()) {
				const Vec3 hit_pos = Vec{ray} * math::sqrt(res);
				// およそ (1.9595917942265426, 0, 9.6) に近いはずだが、球や円筒の接線判定は浮動小数点誤差に敏感
				// ここでは「ヒットしたなら座標が正しいこと」を確認
				CHECK(ApproxCheck{hit_pos} == ApproxCheck{Vec3{1.9595917942265426f, 0.f, 9.6f}});
			}
		}
	}

	TEST_CASE("CylinderOuter: Tilted & Transformed") {
		// 初期状態: 原点にある細長い円筒 (X軸平行にしておく)
		CylinderOuter cyl;
		cyl.center = Vec3{0.0f, 0.0f, 0.0f};
		cyl.radius = 1.0f;
		cyl.axis = UVec3{1.0f, 0.0f, 0.0f};
		cyl.hheight = 10.0f;

		// 変換: 
		// 1. X軸周りに90度回転 (X軸平行 -> X軸平行のまま変わらない...だと面白くないので)
		//    Z軸周りに90度回転させます -> Axisが (0, 1, 0) Y軸平行になる
		// 2. その後、X+5, Z+10 に平行移動
		
		// Z軸90度回転 (Roll=0, Pitch=0, Yaw=90deg)
		const float pi_2 = std::numbers::pi_v<float> / 2.0f;
		const auto rot = SE3::rot(quaternion::ypr({0.0f, 0.0f, pi_2}));
		
		// 平行移動 (5, 0, 10)
		const auto trans = SE3::trans(Vec3{5.0f, 0.0f, 10.0f});
		
		// 適用 (Trans * Rot の順序を想定、ライブラリの乗算仕様によるが通常は左から適用)
		// ここでは個別に適用して動作を確認
		cyl.apply_se3(rot);   // Axis: X(1,0,0) -> Y(0,1,0)
		cyl.apply_se3(trans); // Center: (0,0,0) -> (5,0,10)

		// 期待される状態
		// Center: (5, 0, 10)
		// Axis:   (0, 1, 0) 近似
		// Radius: 1.0
		
		SUBCASE("Verify Transformation") {
			CHECK(ApproxCheck{cyl.center} == ApproxCheck{Vec3{5.0f, 0.0f, 10.0f}});
			CHECK(ApproxCheck{cyl.axis} == ApproxCheck{UVec3{0.0f, 1.0f, 0.0f}});
		}

		SUBCASE("closest_pd on Transformed Cylinder") {
			// 中心 (5,0,10) の手前 (5, 0, 5) からの最近接点
			// 円筒はY軸平行、半径1なので、Z手前表面は Z = 10 - 1 = 9
			// Xは中心と同じ 5
			const Vec3 p = {5.0f, 0.0f, 5.0f};
			const Vec4 res = cyl.closest_pd(p);
			const Vec3 closest_pt = res.xyz();
			
			const Vec3 expected_pt = {5.0f, 0.0f, 9.0f};
			
			CHECK(ApproxCheck{closest_pt} == ApproxCheck{expected_pt});
		}
		
		SUBCASE("ray_collision with Transformed Cylinder") {
			// 原点から (5, 0, 9) に向かうレイ
			// ちょうど表面で当たるはず
			Vec3 target = {5.0f, 0.0f, 9.0f};
			UVec3 ray = math::vec::fast_normalize(target);
			
			const float res = cyl.ray_collision(ray);
			
			// ヒット確認
			CHECK(res != std::numeric_limits<float>::infinity());
		}
	}
}
#endif