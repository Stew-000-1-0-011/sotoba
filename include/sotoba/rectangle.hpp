#pragma once

#include <limits>
#include "fundamental.hpp"
#include "surface.hpp"

namespace sotoba {
	struct Rectangle final {
		Vec4 center; // .xyz = center, .w = 0
		UVec4 u_axis_and_hlen; // .xyz = u_axis (単位ベクトル), .w = half_u_len
		UVec4 v_axis_and_hlen; // .xyz = v_axis (単位ベクトル), .w = half_v_len
		UVec4 normal; // .xyz = normal (単位ベクトル), .w = 0

		auto closest_point_and_distance(this const Rectangle& self, const Vec4& p) noexcept
			-> Vec4 {
			constexpr float visibity_threshold_sq = 0.25f; // cos(pi / 3)^2

			const auto cp = p - self.center;
			// Project d onto the rectangle's local axes
			float u_dist = sycl::dot(cp.xyz(), self.u_axis_and_hlen.xyz());
			float v_dist = sycl::dot(cp.xyz(), self.v_axis_and_hlen.xyz());
			// Clamp the distances to the rectangle's extents
			u_dist = sycl::clamp(u_dist, -self.u_axis_and_hlen.w(), self.u_axis_and_hlen.w());
			v_dist = sycl::clamp(v_dist, -self.v_axis_and_hlen.w(), self.v_axis_and_hlen.w());
			// The closest point is the center plus the clamped projections
			const auto q = self.center.xyz() + u_dist * self.u_axis_and_hlen.xyz()
				+ v_dist * self.v_axis_and_hlen.xyz();
			const auto q_sq = sycl::dot(q, q);
			const auto dot = sycl::dot(-q, Vec3{self.normal.xyz()});
			// qの原点からの可視性チェック
			// 見えるはずのない点には距離無限で返す
			if (dot < 0.0f || dot * dot <= visibity_threshold_sq * q_sq) {
				return {Vec3{}, std::numeric_limits<float>::infinity()};
			}
			const auto qp = p.xyz() - q;
			return {q.xyz(), sycl::dot(qp, qp)};
		}

		void apply_homogeneous(this Rectangle& self, const SE3& h) noexcept {
			self.center = h.trans(self.center);
			self.u_axis_and_hlen = {
				h.only_rot(self.u_axis_and_hlen).xyz(),
				self.u_axis_and_hlen.w()
			};
			self.v_axis_and_hlen = {
				h.only_rot(self.v_axis_and_hlen).xyz(),
				self.v_axis_and_hlen.w()
			};
			self.normal = h.only_rot(self.normal);
		}

		auto ray_collision(this const Rectangle& self, const Ray& ray) noexcept -> Vec4 {
			const auto normal_vec = self.normal.xyz();
			const float denom = sycl::dot(Vec3{normal_vec}, Vec3{ray.direction.xyz()});

			// レイが平面と平行な場合 (内積がほぼ0)
			if (sycl::fabs(denom) < epsilon) {
				return {Vec3{}, std::numeric_limits<float>::infinity()};
			}

			const auto p_to_origin = self.center.xyz() - ray.origin.xyz();
			const float t = sycl::dot(Vec3{p_to_origin}, Vec3{normal_vec}) / denom;

			// 交点がレイの始点より後ろにある場合
			if (t < epsilon) { return {Vec3{}, std::numeric_limits<float>::infinity()}; }

			const auto intersection_point = ray.origin.xyz() + t * ray.direction.xyz();
			const auto vec_from_center = intersection_point - self.center.xyz();

			// 交点が長方形の範囲内にあるかチェック
			const float u_dist = sycl::dot(vec_from_center, Vec3{self.u_axis_and_hlen.xyz()});
			const float v_dist = sycl::dot(vec_from_center, Vec3{self.v_axis_and_hlen.xyz()});

			if (sycl::fabs(u_dist) <= self.u_axis_and_hlen.w()
				&& sycl::fabs(v_dist) <= self.v_axis_and_hlen.w()) {
				const float dist_sq = sycl::dot(intersection_point, intersection_point);
				return {intersection_point, dist_sq};
			}

			// 長方形の範囲外
			return {Vec3{}, std::numeric_limits<float>::infinity()};
		}
	};

	static_assert(surface<Rectangle>);
} // namespace sotoba

#ifdef sotoba_ENABLE_TESTING
	#include <numbers>
	#include "doctest.h"
	#include "fundamental.hpp"

TEST_SUITE("rectangle") {
	using namespace sotoba;
	using namespace sotoba::test;

	TEST_CASE("Rectangle methods") {
		// XY平面上にあり、原点側を向いている長方形をデフォルトとして使用
		Rectangle rect = {
			.center = {0.f, 0.f, 5.f, 0.f},
			.u_axis_and_hlen = UVec4{1.f, 0.f, 0.f, 2.0f}, // 横幅4.0
			.v_axis_and_hlen = UVec4{0.f, 1.f, 0.f, 3.0f}, // 高さ6.0
			.normal = UVec4{0.f, 0.f, -1.f, 0.f}
		};

		// --- closest_point_and_distance のテスト ---
		SUBCASE("closest_point: point is inside the rectangle's projection") {
			const Vec4 p = {1.f, 1.f, 0.f, 0.f}; // Z=0にある点
			const auto result = rect.closest_point_and_distance(p);
			const Vec4 expected_point = {1.f, 1.f, 5.f, 0.f};
			const auto expected = Vec4{expected_point.xyz(), 25.f};
			CHECK(Check{result} == Check{expected}); // 距離5^2 = 25
		}

		SUBCASE("closest_point: point is outside along u-axis") {
			const Vec4 p = {3.f, 1.f, 0.f, 0.f};
			const auto result = rect.closest_point_and_distance(p);
			const Vec4 expected_point = {2.f, 1.f, 5.f, 0.f}; // u-axisの端(2.0f)にクランプされる
			const float expected_dist_sq = (3 - 2) * (3 - 2) + (1 - 1) * (1 - 1)
				+ (0 - 5) * (0 - 5); // 1^2 + 0^2 + (-5)^2 = 26
			const auto expected = Vec4{expected_point.xyz(), expected_dist_sq};
			CHECK(Check{result} == Check{expected});
		}

		SUBCASE("closest_point: point is at a corner") {
			const Vec4 p = {-3.f, -4.f, 0.f, 0.f};
			const auto result = rect.closest_point_and_distance(p);
			const Vec4 expected_point = {-2.f, -3.f, 5.f, 0.f}; // 角(-2, -3)にクランプ
			const float expected_dist_sq = (-3 - (-2)) * (-3 - (-2)) + (-4 - (-3)) * (-4 - (-3))
				+ (0 - 5) * (0 - 5); // (-1)^2 + (-1)^2 + (-5)^2 = 27
			const auto expected = Vec4{expected_point.xyz(), expected_dist_sq};
			CHECK(Check{result} == Check{expected});
		}

		SUBCASE("closest_point: not visible from origin") {
			Rectangle rect_facing_away = rect;
			rect_facing_away.normal = UVec4{0.f, 0.f, 1.f, 0.f}; // 原点と反対側を向いている
			const Vec4 p = {0.f, 0.f, 0.f, 0.f};
			const auto result = rect_facing_away.closest_point_and_distance(p);
			CHECK(result.w() == std::numeric_limits<float>::infinity());
		}

		// --- apply_homogeneous のテスト ---
		SUBCASE("apply_homogeneous: translation") {
			const auto h = SE3{UnitQuaternion::one(), Vec4{10.f, 20.f, 30.f, 0.0f}};
			rect.apply_homogeneous(h);
			const auto expected_center = Vec4{10.f, 20.f, 35.f, 0.f};
			CHECK(Check{rect.center} == Check{expected_center});
			const auto expected_u_axis_and_hlen = UVec4{1.f, 0.f, 0.f, 2.0f};
			CHECK(Check{rect.u_axis_and_hlen} == Check{expected_u_axis_and_hlen}); // 軸は変わらない
			const auto expected_normal = UVec4{0.f, 0.f, -1.f, 0.f};
			CHECK(Check{rect.normal} == Check{expected_normal}); // 法線は変わらない
		}

		SUBCASE("apply_homogeneous: rotation") {
			const float angle90 = std::numbers::pi_v<float> / 2.0f;
			const auto h = SE3::ide().rot_rpy({0.0f, 0.0f, angle90}); // Z軸周りに90度回転
			rect.apply_homogeneous(h);

			const auto e_center = Vec4{0.f, 0.f, 5.f, 0.f};
			const auto e_u_axis_and_hlen = UVec4{0.f, 1.f, 0.f, 2.0f};
			const auto e_v_axis_and_hlen = UVec4{-1.f, 0.f, 0.f, 3.0f};
			const auto e_normal = UVec4{0.f, 0.f, -1.f, 0.f};
			CHECK(Check{rect.center} == Check{e_center}); // Z軸中心なのでcenterは不変
			CHECK(Check{rect.u_axis_and_hlen} == Check{e_u_axis_and_hlen}); // u-axis(X)がY軸方向へ
			CHECK(Check{rect.v_axis_and_hlen} == Check{e_v_axis_and_hlen}); // v-axis(Y)が-X軸方向へ
			CHECK(Check{rect.normal} == Check{e_normal}); // Z軸法線は不変
		}

		// --- ray_collision のテスト ---
		SUBCASE("ray_collision: hit center") {
			const Ray ray = {{0.f, 0.f, 1.f, 0.f}, {0.f, 0.f, 0.f, 0.f}}; // 原点からZ+方向へ
			const auto result = rect.ray_collision(ray);
			const auto expected = Vec4{0.f, 0.f, 5.f, 25.f};
			CHECK(Check{result} == Check{expected}); // z=5でヒット、距離5^2
		}

		SUBCASE("ray_collision: hit off-center") {
			const Ray ray = {{0.f, 0.f, 1.f, 0.f}, {1.f, -2.f, 0.f, 0.f}}; // (1, -2, 0)からZ+方向へ
			const auto result = rect.ray_collision(ray);
			const Vec4 expected_point = {1.f, -2.f, 5.f, 0.f};
			const float dist_sq = 1 * 1 + (-2) * (-2) + 5 * 5; // 1+4+25=30
			const auto expected = Vec4{expected_point.xyz(), dist_sq};
			CHECK(Check{result} == Check{expected});
		}

		SUBCASE("ray_collision: miss (outside bounds)") {
			const Ray ray = {{0.f, 0.f, 1.f, 0.f}, {3.f, 3.f, 0.f, 0.f}}; // u=3, v=3 で範囲外
			const auto result = rect.ray_collision(ray);
			CHECK(result.w() == std::numeric_limits<float>::infinity());
		}

		SUBCASE("ray_collision: miss (parallel ray)") {
			const Ray ray = {{1.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}}; // X軸方向のレイ
			const auto result = rect.ray_collision(ray);
			CHECK(result.w() == std::numeric_limits<float>::infinity());
		}

		SUBCASE("ray_collision: miss (behind ray origin)") {
			const Ray ray = {{0.f, 0.f, 1.f, 0.f}, {0.f, 0.f, 6.f, 0.f}}; // 長方形の奥から発射
			const auto result = rect.ray_collision(ray);
			CHECK(result.w() == std::numeric_limits<float>::infinity());
		}
	}
}
#endif // SOTOBA_ENABLE_TESTING