#pragma once

#include <sycl/sycl.hpp>

namespace sotoba {
	inline constexpr float epsilon = 1e-6f;

	using u32 = unsigned int;
	using i32 = int;

	using Vec3 = sycl::float3;
	using UVec3 = Vec3; // invariant: |v| = 1
	using Vec4 = sycl::float4;
	using UVec4 = Vec4; // invariant: |v.xyz()| = 1

	struct UnitQuaternion final {
		sycl::float4 v; // x, y, z, w

		static constexpr auto from_rpy(const Vec3 rpy) noexcept -> UnitQuaternion {
			const auto half = rpy * 0.5f;
			const auto r = half.x();
			const auto p = half.y();
			const auto y = half.z();

			const auto cr = sycl::cos(r);
			const auto sr = sycl::sin(r);
			const auto cp = sycl::cos(p);
			const auto sp = sycl::sin(p);
			const auto cy = sycl::cos(y);
			const auto sy = sycl::sin(y);

			return UnitQuaternion{
				{sr * cp * cy + cr * sp * sy,
				 cr * sp * cy - sr * cp * sy,
				 cr * cp * sy + sr * sp * cy,
				 cr * cp * cy - sr * sp * sy}
			};
		}

		static constexpr auto zero() -> UnitQuaternion {
			return {{0.0f, 0.0f, 0.0f, 0.0f}};
		}

		static constexpr auto one() -> UnitQuaternion {
			return {{0.0f, 0.0f, 0.0f, 1.0f}};
		}

		friend constexpr auto operator*(const UnitQuaternion& p, const UnitQuaternion& q) noexcept
			-> UnitQuaternion {
			return UnitQuaternion{
				{p.v.w() * q.v.x() + p.v.x() * q.v.w() + p.v.y() * q.v.z() - p.v.z() * q.v.y(),
				 p.v.w() * q.v.y() - p.v.x() * q.v.z() + p.v.y() * q.v.w() + p.v.z() * q.v.x(),
				 p.v.w() * q.v.z() + p.v.x() * q.v.y() - p.v.y() * q.v.x() + p.v.z() * q.v.w(),
				 p.v.w() * q.v.w() - p.v.x() * q.v.x() - p.v.y() * q.v.y() - p.v.z() * q.v.z()}
			};
		}

		constexpr auto operator!(this const UnitQuaternion& self) noexcept -> UnitQuaternion {
			return {{-self.v.xyz(), self.v.w()}};
		}

		constexpr auto rot_vec(this const UnitQuaternion& self, const Vec4& v) noexcept -> Vec4 {
			const auto vq = UnitQuaternion{{v.xyz(), 0.0}};
			return (self * vq * !self).v;
		}

		constexpr auto normalize(this const UnitQuaternion& self) noexcept -> UnitQuaternion {
			const auto length = sycl::length(self.v);
			if (length < epsilon) return self;
			else return UnitQuaternion{self.v / length};
		}
	};

	// 3x3 行列 (共分散行列の計算と結果に使用)
	struct Mat3d final {
		std::array<float, 9> v{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

		// Mat3d型の加算演算子 (sycl::reductionでカスタム型を使うために必要)
		constexpr friend auto operator+(const Mat3d& a, const Mat3d& b) noexcept -> Mat3d {
			Mat3d result;
			for (i32 i = 0; i < 9; ++i) result.v[i] = a.v[i] + b.v[i];
			return result;
		}
	};

	// 剛体変換
	struct SE3 final {
		UnitQuaternion q;
		Vec4 p;

		static constexpr auto ide() noexcept -> SE3 {
			return SE3{UnitQuaternion::one(), Vec4{}};
		}

		constexpr auto trans(this const SE3& self, const Vec4& v) noexcept -> Vec4 {
			return self.q.rot_vec(v) + self.p;
		}

		constexpr auto only_rot(this const SE3& self, const UVec4& v) noexcept -> UVec4 {
			return self.q.rot_vec(v);
		}

		constexpr auto normalize(this const SE3& self) noexcept -> SE3 {
			return SE3{self.q.normalize(), Vec4{self.p.xyz(), 0.f}};
		}

		constexpr auto rot_rpy(this const SE3& self, const Vec3& rpy) noexcept -> SE3 {
			return SE3{UnitQuaternion::from_rpy(rpy), Vec4{}} * self;
		}

		constexpr auto operator!(this const SE3& self) noexcept -> SE3 {
			const auto conj_q = !self.q;
			return SE3{conj_q, -conj_q.rot_vec(self.p)};
		}

		constexpr friend auto operator*(const SE3& l, const SE3& r) -> SE3 {
			return SE3{l.q * r.q, l.p + l.q.rot_vec(r.p)};
		}
	};

	struct Ray final {
		UVec4 direction;
		Vec4 origin;
	};
} // namespace sotoba

#ifdef sotoba_ENABLE_TESTING
	#include <iostream>
	#include <numbers>
	#include "doctest.h"

namespace sotoba::test {
	template <class T_>
	struct Check final {
		static_assert(false, "No specialization found.");
	};

	constexpr auto approx_eps = static_cast<double>(std::numeric_limits<float>::epsilon()) * 100;

	template <>
	struct Check<sotoba::Vec4> final {
		sotoba::Vec4 v;
		float eps{approx_eps};

		friend constexpr auto operator==(const Check& l, const Check& r) noexcept -> bool {
			const auto eps = std::max(l.eps, r.eps);
			return l.v.x() == doctest::Approx(r.v.x()).epsilon(eps)
				&& l.v.y() == doctest::Approx(r.v.y()).epsilon(eps)
				&& l.v.z() == doctest::Approx(r.v.z()).epsilon(eps)
				&& l.v.w() == doctest::Approx(r.v.w()).epsilon(eps);
		}

		friend auto operator<<(std::ostream& os, const Check& v) -> std::ostream& {
			return os << "{ " << v.v.x() << ' ' << v.v.y() << ' ' << v.v.z() << ' ' << v.v.w()
					  << " }";
		}
	};

	template <>
	struct Check<UnitQuaternion> final {
		sotoba::UnitQuaternion q;
		float eps{approx_eps};

		friend constexpr auto operator==(const Check& l, const Check& r) noexcept -> bool {
			const auto eps = std::max(l.eps, r.eps);
			return doctest::Approx(sycl::fabs(sycl::dot(l.q.v, r.q.v))).epsilon(eps) == 1.0f;
		}

		friend auto operator<<(std::ostream& os, const Check& v) -> std::ostream& {
			return os << Check<Vec4>{v.q.v};
		}
	};

	template <>
	struct Check<SE3> final {
		SE3 f;
		float eps{approx_eps};

		friend constexpr auto operator==(const Check& l, const Check& r) noexcept -> bool {
			return Check<UnitQuaternion>{l.f.q, l.eps} == Check<UnitQuaternion>{r.f.q, r.eps}
			&& Check<Vec4>{l.f.p, l.eps} == Check<Vec4>{r.f.p, r.eps};
		}

		friend auto operator<<(std::ostream& os, const Check& f) -> std::ostream& {
			return os << Check<UnitQuaternion>{f.f.q} << ", " << Check<Vec4>{f.f.p};
		}
	};

	template <class T_>
	Check(const T_&) -> Check<T_>;

	template <class T_>
	Check(const T_&, const float) -> Check<T_>;
} // namespace sotoba::test

TEST_SUITE("fundamental") {
	using namespace sotoba;
	using namespace sotoba::test;

	TEST_CASE("UnitQuaternion tests") {
		const auto q_identity = UnitQuaternion::one();

		// Z軸周りに90度回転するクォータニオン
		const float angle90 = std::numbers::pi / 2.0f;
		const auto q_z90 =
			UnitQuaternion{{0.f, 0.f, std::sin(angle90 / 2.f), std::cos(angle90 / 2.f)}}.normalize(
			);

		// X軸周りに180度回転するクォータニオン
		const float angle180 = std::numbers::pi;
		const auto q_x180 = UnitQuaternion{
			{std::sin(angle180 / 2.f), 0.f, 0.f, std::cos(angle180 / 2.f)}
		}.normalize();

		SUBCASE("from_rpy (Roll, Pitch, Yaw)") {
			const float angle90 = std::numbers::pi_v<float> / 2.0f;
			const float c45 = std::cos(angle90 / 2.0f); // cos(45)
			const float s45 = std::sin(angle90 / 2.0f); // sin(45)

			// --- Test Case: Identity ---
			// Input
			const Vec3 rpy_identity = {0.f, 0.f, 0.f};
			// Expected: q_identityはテストケースの冒頭で定義済み
			// Action
			const auto result_identity = UnitQuaternion::from_rpy(rpy_identity);
			// Check
			CHECK(Check{result_identity} == Check{UnitQuaternion::one()});

			// --- Test Case: Pure Yaw ---
			// Input
			const Vec3 rpy_yaw = {0.f, 0.f, angle90};
			// Expected
			const Vec4 expected_yaw = {0.f, 0.f, s45, c45};
			// Action
			const auto result_yaw = UnitQuaternion::from_rpy(rpy_yaw);
			// Check
			CHECK(Check{result_yaw.v} == Check{expected_yaw});

			// --- Test Case: Pure Pitch ---
			// Input
			const Vec3 rpy_pitch = {0.f, angle90, 0.f};
			// Expected
			const Vec4 expected_pitch = {0.f, s45, 0.f, c45};
			// Action
			const auto result_pitch = UnitQuaternion::from_rpy(rpy_pitch);
			// Check
			CHECK(Check{result_pitch.v} == Check{expected_pitch});

			// --- Test Case: Pure Roll ---
			// Input
			const Vec3 rpy_roll = {angle90, 0.f, 0.f};
			// Expected
			const Vec4 expected_roll = {s45, 0.f, 0.f, c45};
			// Action
			const auto result_roll = UnitQuaternion::from_rpy(rpy_roll);
			// Check
			CHECK(Check{result_roll.v} == Check{expected_roll});

			// --- Test Case: Combined Pitch and Yaw ---
			// Input
			const Vec3 rpy_pitch_yaw = {0.f, angle90, angle90};
			// Expected: Z軸(Yaw)回転 → Y軸(Pitch)回転 の順で適用
			const auto q_pitch_manual = UnitQuaternion{{0.f, s45, 0.f, c45}};
			const auto q_yaw_manual = UnitQuaternion{{0.f, 0.f, s45, c45}};
			const auto expected_pitch_yaw = q_pitch_manual * q_yaw_manual;
			// Action
			const auto result_pitch_yaw = UnitQuaternion::from_rpy(rpy_pitch_yaw);
			// Check
			CHECK(Check{result_pitch_yaw.v} == Check{expected_pitch_yaw.v});
		}

		SUBCASE("Multiplication with identity") {
			const auto result = q_z90 * q_identity;
			CHECK(Check{result.v} == Check{q_z90.v});
		}

		SUBCASE("Multiplication") {
			// Z軸90度回転を2回合成するとZ軸180度回転になる
			const auto result = q_z90 * q_z90;
			const auto q_z180 = UnitQuaternion{
				{0.f, 0.f, std::sin(angle180 / 2.f), std::cos(angle180 / 2.f)}
			}.normalize();
			CHECK(Check{result.v} == Check{q_z180.v});
		}

		SUBCASE("Conjugate (Inverse)") {
			const auto result = q_x180 * !q_x180;
			CHECK(Check{result.v} == Check{q_identity.v});
		}

		SUBCASE("Vector rotation") {
			const Vec4 v_x = {1.f, 0.f, 0.f, 0.f};
			const Vec4 expected_y = {0.f, 1.f, 0.f, 0.f};

			// X軸ベクトルをZ軸周りに90度回転させるとY軸ベクトルになる
			const auto rotated_v = q_z90.rot_vec(v_x);
			CHECK(Check{rotated_v} == Check{expected_y});
		}

		SUBCASE("Normalization") {
			auto non_unit_q = UnitQuaternion{{1.f, 2.f, 3.f, 4.f}};
			auto normalized_q = non_unit_q.normalize();
			CHECK(sycl::length(normalized_q.v) == doctest::Approx(1.0f));
		}
	}

	TEST_CASE("SE3 tests") {
		const auto q_identity = UnitQuaternion::one();
		const Vec4 p_zero = {};
		const auto se3_identity = SE3::ide();

		// Z軸周りに90度回転
		const float angle90 = std::numbers::pi / 2.0f;
		const auto q_z90 =
			UnitQuaternion{{0.f, 0.f, std::sin(angle90 / 2.f), std::cos(angle90 / 2.f)}}.normalize(
			);
		const auto se3_rot_z90 = SE3{q_z90, p_zero};

		// (5, 0, 0) 方向への移動
		const auto p_trans_x5 = Vec4{5.f, 0.f, 0.f, 0.f};
		const auto se3_trans_x5 = SE3{q_identity, p_trans_x5};

		const Vec4 test_vec = {1.f, 2.f, 3.f, 0.f};

		SUBCASE("Identity transformation") {
			const auto result = se3_identity.trans(test_vec);
			CHECK(Check{result} == Check{test_vec});
		}

		SUBCASE("Composition") {
			// T2(v) = Rot(v), T1(v) = Trans(v)
			// (T2 * T1)(v) = T2(T1(v))
			const auto composed = se3_rot_z90 * se3_trans_x5;

			// T1 -> T2 の順で適用
			const auto sequential_apply = se3_rot_z90.trans(se3_trans_x5.trans(test_vec));

			// 合成した変換を一回で適用
			const auto composed_apply = composed.trans(test_vec);

			CHECK(Check{sequential_apply} == Check{composed_apply});
		}

		SUBCASE("Inverse") {
			// T = RotZ90 + TransX5
			const auto T = SE3{q_z90, p_trans_x5};
			const auto T_inv = !T;

			// T^-1 * T は単位変換になるはず
			const auto should_be_identity = T_inv * T;
			CHECK(Check{should_be_identity.q.v} == Check{q_identity.v});
			CHECK(Check{should_be_identity.p} == Check{p_zero});

			// T(v)を計算し、その結果にT^-1を適用すると元のvに戻る
			const auto transformed_vec = T.trans(test_vec);
			const auto restored_vec = T_inv.trans(transformed_vec);
			CHECK(Check{restored_vec} == Check{test_vec});
		}

		SUBCASE("Translation only") {
			const auto result = se3_trans_x5.trans(test_vec);
			const auto expected = test_vec + p_trans_x5;
			CHECK(Check{result} == Check{expected});
		}

		SUBCASE("Rotation only") {
			const auto result = se3_rot_z90.trans(test_vec);
			const auto expected = q_z90.rot_vec(test_vec);
			CHECK(Check{result} == Check{expected});
		}
	}
}
#endif // SOTOBA_ENABLE_TESTS