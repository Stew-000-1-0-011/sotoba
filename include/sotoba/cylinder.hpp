#pragma once

#include <limits>
#include "fundamental.hpp"
#include "surface.hpp"

namespace sotoba {
	// 上下の底面が無い円筒
	struct CylinderOuter final {
		Vec4 center_and_radius; // .xyz = center, .w = radius
		UVec4 axis_and_hheight; // .xyz = axis (単位ベクトル), .w = 円柱高さ / 2

		auto closest_point_and_distance(this const CylinderOuter& self, const Vec4& p) noexcept
			-> Vec4 {
			static constexpr auto visible_threshold_sq = 0.25f; // cos(pi / 3)^2

			/*
			c: center
			d: 中心軸(有限長さ)上のpの最近接点
			e: 中心軸(無限長さ)上のpの最近接点
			q: 円柱(有限長さ)側面上の最近接点
			*/
			const auto axis = self.axis_and_hheight.xyz();
			const auto hheight = self.axis_and_hheight.w();
			const auto c = self.center_and_radius.xyz();
			const auto r = self.center_and_radius.w();
			const auto cp = p.xyz() - c;
			const auto vertical = sycl::dot(Vec3{cp}, Vec3{axis});
			const auto cd = sycl::clamp(vertical, -hheight, hheight) * axis;
			const auto ce = vertical * axis;
			const auto ep = cp - ce;
			if (sycl::dot(ep, ep) < epsilon) {
				return Vec4{Vec3{}, std::numeric_limits<float>::infinity()};
			}
			const auto dq = r * sycl::fast_normalize(ep);
			const auto q = c + cd + dq;

			// qの可視性チェック
			// 原点からqが見えるか
			const auto dot = sycl::dot(-q, dq);
			const auto q_sq = sycl::dot(q, q);
			const auto dq_sq = r * r;
			if (dot < 0.0f || dot * dot < visible_threshold_sq * q_sq * dq_sq) {
				return Vec4{Vec3{}, std::numeric_limits<float>::infinity()};
			}
			const auto pq = q - p.xyz();
			return Vec4{q, sycl::dot(pq, pq)};
		}

		void apply_homogeneous(this CylinderOuter& self, const SE3& h) noexcept {
			const auto center = h.trans(Vec4{self.center_and_radius.xyz(), 0.0f}).xyz();
			const auto axis = h.only_rot(self.axis_and_hheight).xyz();
			self.center_and_radius = {center, self.center_and_radius.w()};
			self.axis_and_hheight = {axis, self.axis_and_hheight.w()};
		}

		auto ray_collision(this const CylinderOuter& self, const Ray& ray) noexcept -> Vec4 {
			const auto ray_dir = ray.direction.xyz();
			const auto ray_origin = ray.origin.xyz();

			const auto axis = self.axis_and_hheight.xyz();
			const auto center = self.center_and_radius.xyz();
			const auto radius = self.center_and_radius.w();
			const auto hheight = self.axis_and_hheight.w();

			const auto co = ray_origin - center;
			const float dot_d_a = sycl::dot(Vec3{ray_dir}, Vec3{axis});
			const float dot_co_a = sycl::dot(Vec3{co}, Vec3{axis});

			const float a = 1.0f - dot_d_a * dot_d_a;
			const float b = 2.0f * (sycl::dot(Vec3{ray_dir}, Vec3{co}) - dot_d_a * dot_co_a);
			const float c = sycl::dot(Vec3{co}, Vec3{co}) - dot_co_a * dot_co_a - radius * radius;

			// レイが軸と平行な場合
			if (sycl::fabs(a) < epsilon) return {Vec3{}, std::numeric_limits<float>::infinity()};

			const float delta = b * b - 4.0f * a * c;

			// 交点がない場合
			if (delta < 0.0f) { return {Vec3{}, std::numeric_limits<float>::infinity()}; }

			const float sqrt_delta = sycl::sqrt(delta);
			float t = (-b - sqrt_delta) / (2.0f * a);

			// 最初の交点が背後にあるか、高さの範囲外かチェック
			if (t < epsilon || sycl::fabs(dot_co_a + t * dot_d_a) > hheight) {
				t = (-b + sqrt_delta) / (2.0f * a);
				// 2番目の交点も背後にあるか、高さの範囲外なら交差しない
				if (t < epsilon || sycl::fabs(dot_co_a + t * dot_d_a) > hheight) {
					return {Vec3{}, std::numeric_limits<float>::infinity()};
				}
			}
			// 円柱の内側の点も出てきてしまうが、まあいいか

			const auto intersection_point = ray_origin + t * ray_dir;
			return {intersection_point, t * t};
		}
	};

	static_assert(surface<CylinderOuter>);
} // namespace sotoba

#ifdef sotoba_ENABLE_TESTING
	#include <numbers>
	#include "doctest.h"
	#include "fundamental.hpp"

/*
可視性チェックにおいては、**観測点は常に原点である**ことに注意！
レイとの交差判定のときだけ原点とレイ源点が異なるので注意
*/

TEST_SUITE("cylinder") {
	using namespace sotoba;
	using namespace sotoba::test;

	TEST_CASE("CylinderOuter::closest_point_and_distance") {
		SUBCASE("原点を包含する円筒は無効値を返す") {
			// この円筒は中心(0,0,0)にあり、原点を包含する
			CylinderOuter cylinder{/*center_and_radius*/ {0.f, 0.f, 0.f, 1.f},
								   /*axis_and_hheight*/ {0.f, 0.f, 1.f, 2.f}};
			const Vec4 p = {2.f, 0.f, 1.f, 0.f};
			const auto result = cylinder.closest_point_and_distance(p);

			// 原点を包含するため、最近接点は無効値(wがinf)になる
			CHECK(result.w() == std::numeric_limits<float>::infinity());
		}

		SUBCASE("原点を包含しない円筒: 点が側面にあり最近接点が計算される") {
			// この円筒は中心(3,0,0)にあり、原点を包含しない
			CylinderOuter cylinder{/*center_and_radius*/ {3.f, 0.f, 0.f, 1.f},
								   /*axis_and_hheight*/ {0.f, 0.f, 1.f, 2.f}};
			const Vec4 p = {1.f, 0.f, 1.f, 0.f};
			const auto result = cylinder.closest_point_and_distance(p);
			const Vec4 expected = {2.f, 0.f, 1.f, 1.f}; // {point, dist_sq}

			CHECK(Check{result} == Check{expected});
		}

		SUBCASE("原点を包含しない円筒: 点が上方にあり最近接点は上端になる") {
			// この円筒は中心(3,0,0)にあり、原点を包含しない
			CylinderOuter cylinder{/*center_and_radius*/ {3.f, 0.f, 0.f, 1.f},
								   /*axis_and_hheight*/ {0.f, 0.f, 1.f, 2.f}};
			const Vec4 p = {1.f, 0.f, 3.f, 0.f};
			const auto result = cylinder.closest_point_and_distance(p);
			const Vec3 expected_point = {2.f, 0.f, 2.f};
			const float expected_dist_sq =
				sycl::dot(expected_point - p.xyz(), expected_point - p.xyz());
			const Vec4 expected = {expected_point, expected_dist_sq};

			CHECK(Check{result} == Check{expected});
		}

		SUBCASE("点が中心軸上にあり、最近接点は定義できない (無限大が返る)") {
			CylinderOuter cylinder{/*center_and_radius*/ {3.f, 0.f, 0.f, 1.f},
								   /*axis_and_hheight*/ {0.f, 1.f, 0.f, 2.f}};
			const Vec4 p = {3.f, 5.f, 0.f, 0.f};
			const auto result = cylinder.closest_point_and_distance(p);
			CHECK(result.w() == std::numeric_limits<float>::infinity());
		}
	}

	TEST_CASE("CylinderOuter::apply_homogeneous") {
		CylinderOuter cylinder{/*center_and_radius*/ {1.f, 2.f, 3.f, 5.f},
							   /*axis_and_hheight*/ {0.f, 0.f, 1.f, 4.f}};

		SUBCASE("並進移動") {
			const SE3 translation{UnitQuaternion::one(), {10.f, 20.f, 30.f, 0.f}};
			cylinder.apply_homogeneous(translation);

			const Vec4 expected_center_and_radius = {11.f, 22.f, 33.f, 5.f};
			const Vec4 expected_axis_and_hheight = {0.f, 0.f, 1.f, 4.f};

			CHECK(Check{cylinder.center_and_radius} == Check{expected_center_and_radius});
			CHECK(Check{cylinder.axis_and_hheight} == Check{expected_axis_and_hheight});
		}

		SUBCASE("回転") {
			const SE3 rotation{
				UnitQuaternion::from_rpy({0.f, static_cast<float>(std::numbers::pi / 2), 0.f}),
				{}
			};
			cylinder.apply_homogeneous(rotation);

			const Vec4 expected_center_and_radius = {3.f, 2.f, -1.f, 5.f};
			const Vec4 expected_axis_and_hheight = {1.f, 0.f, 0.f, 4.f};

			CHECK(Check{cylinder.center_and_radius} == Check{expected_center_and_radius});
			CHECK(Check{cylinder.axis_and_hheight} == Check{expected_axis_and_hheight});
		}
	}

	TEST_CASE("CylinderOuter::ray_collision") {
		CylinderOuter cylinder{/*center_and_radius*/ {0.f, 0.f, 0.f, 1.f},
							   /*axis_and_hheight*/ {0.f, 0.f, 1.f, 2.f}};

		SUBCASE("レイが円筒を貫通する") {
			const Ray ray{{1.f, 0.f, 0.f, 0.f}, {-2.f, 0.f, 0.f, 0.f}};
			const auto result = cylinder.ray_collision(ray);
			const Vec4 expected = {-1.f, 0.f, 0.f, 1.f}; // {point, dist_sq}

			CHECK(Check{result} == Check{expected});
		}

		SUBCASE("レイが円筒と交差しない") {
			const Ray ray{{1.f, 0.f, 0.f, 0.f}, {-2.f, 2.f, 0.f, 0.f}};
			const auto result = cylinder.ray_collision(ray);
			CHECK(result.w() == std::numeric_limits<float>::infinity());
		}

		SUBCASE("レイが円筒に接する") {
			const Ray ray{{1.f, 0.f, 0.f, 0.f}, {-2.f, 1.f, 0.f, 0.f}};
			const auto result = cylinder.ray_collision(ray);
			const Vec3 expected_point = {0.f, 1.f, 0.f};
			const float expected_dist_sq =
				sycl::dot(expected_point - ray.origin.xyz(), expected_point - ray.origin.xyz());
			const Vec4 expected = {expected_point, expected_dist_sq};

			CHECK(Check{result} == Check{expected});
		}

		SUBCASE("レイの始点が円筒の内部にある") {
			const Ray ray{{1.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
			const auto result = cylinder.ray_collision(ray);
			const Vec4 expected = {1.f, 0.f, 0.f, 1.f};

			CHECK(Check{result} == Check{expected});
		}
	}
}
#endif