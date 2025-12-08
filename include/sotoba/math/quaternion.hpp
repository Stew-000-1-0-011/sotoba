#pragma once

#include <cmath>
#include <format>

#include "vec.hpp"

namespace sotoba::math::quaternion_impl {
	template <bool is_unit_>
	struct Quaternion final {
		Vec4 v; // x, y, z, w

		constexpr operator Quaternion<false>() noexcept {
			return Quaternion<false>{this->v};
		}

		template <bool is_unit2_>
		friend constexpr auto
		operator*(const Quaternion& p, const Quaternion<is_unit2_>& q) noexcept
			-> Quaternion<is_unit_ && is_unit2_> {
			return Quaternion<is_unit_ && is_unit2_>{
				{p.v.w() * q.v.x() + p.v.x() * q.v.w() + p.v.y() * q.v.z() - p.v.z() * q.v.y(),
				 p.v.w() * q.v.y() - p.v.x() * q.v.z() + p.v.y() * q.v.w() + p.v.z() * q.v.x(),
				 p.v.w() * q.v.z() + p.v.x() * q.v.y() - p.v.y() * q.v.x() + p.v.z() * q.v.w(),
				 p.v.w() * q.v.w() - p.v.x() * q.v.x() - p.v.y() * q.v.y() - p.v.z() * q.v.z()}
			};
		}

		constexpr auto conj() const noexcept -> Quaternion {
			return {Vec4{-this->v.xyz(), Vec{this->v.w()}}};
		}

		constexpr friend auto rot_vec(const Quaternion& q, const Vec3& v) noexcept -> Vec3
			requires is_unit_
		{
			const auto vq = Quaternion<false>{{v, Vec{0.f}}};
			return (q * vq * q.conj()).v.xyz();
		}

		constexpr friend auto rot_uvec(const Quaternion& q, const UVec3& v) noexcept -> UVec3
			requires is_unit_
		{
			const auto vq = Quaternion<true>{{v, Vec{0.f}}};
			return vec::as_uvec(Vec3{(q * vq * q.conj()).v.xyz()});
		}

		constexpr auto fast_length() const noexcept -> float {
			return vec::fast_length(this->v);
		}

		constexpr auto normalize() noexcept -> Quaternion<true> {
			return Quaternion<true>{vec::fast_normalize(this->v)};
		}
	};

	// Yaw -> Pitch -> Roll の順に適用
	inline constexpr auto ypr(const Vec3 ypr) noexcept -> Quaternion<true> {
		const auto half = ypr * 0.5f;
		const auto r = half.x();
		const auto p = half.y();
		const auto y = half.z();

		const auto cr = std::cos(r);
		const auto sr = std::sin(r);
		const auto cp = std::cos(p);
		const auto sp = std::sin(p);
		const auto cy = std::cos(y);
		const auto sy = std::sin(y);

		return Quaternion<true>{
			{sr * cp * cy + cr * sp * sy,
			 cr * sp * cy - sr * cp * sy,
			 cr * cp * sy + sr * sp * cy,
			 cr * cp * cy - sr * sp * sy}
		};
	}

	// Roll -> Pitch -> Yaw の順に適用
	inline constexpr auto rpy(const Vec3 rpy) noexcept -> Quaternion<true> {
		const auto half = rpy * 0.5f;
		const auto r = half.x();
		const auto p = half.y();
		const auto y = half.z();

		const auto cr = std::cos(r);
		const auto sr = std::sin(r);
		const auto cp = std::cos(p);
		const auto sp = std::sin(p);
		const auto cy = std::cos(y);
		const auto sy = std::sin(y);

		return Quaternion<true>{
			{sr * cp * cy - cr * sp * sy,
			 cr * sp * cy + sr * cp * sy,
			 cr * cp * sy - sr * sp * cy,
			 cr * cp * cy + sr * sp * sy}
		};
	}

	inline constexpr auto ide() noexcept -> Quaternion<true> {
		return {{0.f, 0.f, 0.f, 1.f}};
	}

	inline constexpr auto to_mat(const Quaternion<true>& q) noexcept -> SquareMat<3> {
		SquareMat<3> ret{};

		const float x = q.v.x();
		const float y = q.v.y();
		const float z = q.v.z();
		const float w = q.v.w();

		const float x2 = x + x;
		const float y2 = y + y;
		const float z2 = z + z;

		const float xx = x * x2;
		const float xy = x * y2;
		const float xz = x * z2;
		const float yy = y * y2;
		const float yz = y * z2;
		const float zz = z * z2;
		const float wx = w * x2;
		const float wy = w * y2;
		const float wz = w * z2;

		ret[0, 0] = 1.0f - (yy + zz);
		ret[0, 1] = xy - wz;
		ret[0, 2] = xz + wy;

		ret[1, 0] = xy + wz;
		ret[1, 1] = 1.0f - (xx + zz);
		ret[1, 2] = yz - wx;

		ret[2, 0] = xz - wy;
		ret[2, 1] = yz + wx;
		ret[2, 2] = 1.0f - (xx + yy);

		return ret;
	}
} // namespace sotoba::math::quaternion_impl

namespace sotoba::math {
	template <bool is_unit_>
	using QuaternionT = quaternion_impl::Quaternion<is_unit_>;
	using UQuaternion = quaternion_impl::Quaternion<true>;
	using Quaternion = quaternion_impl::Quaternion<false>;

	namespace quaternion {
		using quaternion_impl::ide;
		using quaternion_impl::rpy;
		using quaternion_impl::to_mat;
		using quaternion_impl::ypr;
	} // namespace quaternion
} // namespace sotoba::math

namespace sotoba {
	template <bool is_unit_>
	struct Repr<math::QuaternionT<is_unit_>> final {
		static auto repr(const math::QuaternionT<is_unit_>& self) -> std::string {
			std::string ret{};
			ret += std::format("Quaternion<{}>{{", is_unit_);
			for (u8 i = 0; i < 4; ++i) { ret += std::format("{}, ", self.v[i]); }
			ret += '}';
			return ret;
		}
	};
} // namespace sotoba

#ifdef sotoba_ENABLE_TESTING
	#include <cmath>
	#include <numbers>

	#include <doctest.h>

	#include "approx_check.hpp"

namespace sotoba::math {
	template <bool u>
	struct ApproxCheckImpl<QuaternionT<u>> {
		static auto
		compare(const QuaternionT<u>& q1, const QuaternionT<u>& q2, const std::optional<float> eps)
			-> bool {
			return ApproxCheckImpl<Vec4>::compare(q1.v, q2.v, eps);
		}
	};
} // namespace sotoba::math

TEST_SUITE("quaternion.hpp") {
	using namespace sotoba::math;

	// テスト用の定数
	using std::numbers::pi;
	constexpr float pi_2 = pi / 2.0f;
	constexpr float epsilon = 1e-5f; // 少し緩めに設定

	TEST_CASE("Construction and Identity") {
		// 単位元 (0, 0, 0, 1)
		auto q_id = quaternion::ide();
		CHECK(q_id.v.x() == 0.f);
		CHECK(q_id.v.y() == 0.f);
		CHECK(q_id.v.z() == 0.f);
		CHECK(q_id.v.w() == 1.f);

		CHECK(q_id.fast_length() == doctest::Approx(1.f));
	}

	TEST_CASE("Conjugate") {
		// q = (1, 2, 3, 4)
		Quaternion q{{1.f, 2.f, 3.f, 4.f}};
		auto q_conj = q.conj();

		// 共役は虚部(xyz)が反転し、実部(w)はそのまま
		CHECK(q_conj.v.x() == -1.f);
		CHECK(q_conj.v.y() == -2.f);
		CHECK(q_conj.v.z() == -3.f);
		CHECK(q_conj.v.w() == 4.f);
	}

	TEST_CASE("Multiplication (Hamilton Product)") {
		// 基底ベクトルの積: i * j = k
		// x=1 (i), y=1 (j), z=1 (k), w=0 (real)
		Quaternion qi{{1.f, 0.f, 0.f, 0.f}};
		Quaternion qj{{0.f, 1.f, 0.f, 0.f}};
		Quaternion qk{{0.f, 0.f, 1.f, 0.f}};

		// qi * qj -> qk
		auto res_k = qi * qj;
		CHECK(ApproxCheck{res_k} == ApproxCheck{qk});

		// qj * qi -> -qk
		auto res_neg_k = qj * qi;
		Quaternion neg_qk{{0.f, 0.f, -1.f, 0.f}};
		CHECK(ApproxCheck{res_neg_k} == ApproxCheck{neg_qk});

		// qi * qi -> -1
		auto res_neg_1 = qi * qi;
		Quaternion neg_one{{0.f, 0.f, 0.f, -1.f}};
		CHECK(ApproxCheck{res_neg_1} == ApproxCheck{neg_one});

		// 単位元との積
		auto q = Quaternion{{1.f, 2.f, 3.f, 4.f}};
		CHECK(ApproxCheck{q * quaternion::ide()} == ApproxCheck{q});
		CHECK(ApproxCheck{quaternion::ide() * q} == ApproxCheck{q});
	}

	TEST_CASE("Normalization") {
		Quaternion q{{1.f, 1.f, 1.f, 1.f}}; // 長さは sqrt(4) = 2
		auto uq = q.normalize();

		// 長さが1になっているか
		CHECK(uq.fast_length() == doctest::Approx(1.f));

		// 値が 0.5 (1/2) になっているか
		CHECK(uq.v.x() == doctest::Approx(0.5f));
		CHECK(uq.v.w() == doctest::Approx(0.5f));

		// 型が UQuaternion (is_unit_ = true) であることの確認
		static_assert(std::is_same_v<decltype(uq), UQuaternion>);
	}

	TEST_CASE("RPY (Euler Angles) Construction") {
		// Roll (X軸回転) 90度
		auto q_roll = quaternion::ypr(Vec3{pi_2, 0.f, 0.f});
		// cos(45) = sin(45) = 0.707...
		// x=sin, w=cos, others=0
		CHECK(q_roll.v.x() == doctest::Approx(std::sin(pi / 4.f)));
		CHECK(q_roll.v.w() == doctest::Approx(std::cos(pi / 4.f)));
		CHECK(q_roll.v.y() == doctest::Approx(0.f).epsilon(epsilon));
		CHECK(q_roll.v.z() == doctest::Approx(0.f).epsilon(epsilon));

		// Yaw (Z軸回転) 90度
		auto q_yaw = quaternion::ypr(Vec3{0.f, 0.f, pi_2});
		CHECK(q_yaw.v.z() == doctest::Approx(std::sin(pi / 4.f)));
		CHECK(q_yaw.v.w() == doctest::Approx(std::cos(pi / 4.f)));
	}

	TEST_CASE("Vector Rotation (rot_vec)") {
		Vec3 v_in{1.f, 0.f, 0.f}; // X軸方向のベクトル

		SUBCASE("Rotate +90 deg around Z-axis (Yaw)") {
			// X軸 -> Y軸 になるはず
			auto q = quaternion::ypr(Vec3{0.f, 0.f, pi_2});
			Vec3 v_out = rot_vec(q, v_in);

			CHECK(v_out.x() == doctest::Approx(0.f).epsilon(epsilon));
			CHECK(v_out.y() == doctest::Approx(1.f).epsilon(epsilon));
			CHECK(v_out.z() == doctest::Approx(0.f).epsilon(epsilon));
		}

		SUBCASE("Rotate +90 deg around Y-axis (Pitch)") {
			// X軸 -> -Z軸 になるはず (右手系)
			auto q = quaternion::ypr(Vec3{0.f, pi_2, 0.f});
			Vec3 v_out = rot_vec(q, v_in);

			CHECK(v_out.x() == doctest::Approx(0.f).epsilon(epsilon));
			CHECK(v_out.y() == doctest::Approx(0.f).epsilon(epsilon));
			CHECK(v_out.z() == doctest::Approx(-1.f).epsilon(epsilon));
		}

		SUBCASE("Identity rotation") {
			auto q = quaternion::ide();
			Vec3 v_out = rot_vec(q, v_in);
			CHECK(ApproxCheck{v_out} == ApproxCheck{v_in});
		}
	}

	TEST_CASE("Unit Vector Rotation (rot_uvec)") {
		UVec3 v_in{1.f, 0.f, 0.f}; // X軸単位ベクトル
		auto q = quaternion::ypr(Vec3{0.f, 0.f, pi_2}); // Z軸90度

		UVec3 v_out = rot_uvec(q, v_in);

		CHECK(v_out.x() == doctest::Approx(0.f).epsilon(epsilon));
		CHECK(v_out.y() == doctest::Approx(1.f).epsilon(epsilon));

		// 戻り値が UVec3 であること
		static_assert(std::is_same_v<decltype(v_out), UVec3>);
	}
}
#endif