#pragma once

#include <cmath>
#include <format>
#include <string>
#include <type_traits>

#include <Eigen/Geometry>
#include <sophus/se3.hpp>

#include "sotoba/math/approx_check.hpp"
#include "sotoba/math/vec_forward_decl.hpp"
#include "sotoba/repr.hpp"

#include "vec.hpp"

namespace sotoba::math {
	static_assert(
		sizeof(Vec3) == sizeof(Eigen::Vector3f) && alignof(Vec3) == alignof(float)
			&& std::is_standard_layout_v<Vec3>,
		"to_eigen は Vec3 が float[3] と同じレイアウトであることに依存する"
	);

	template <bool is_unit_>
	inline auto to_eigen(const Vec<3, is_unit_>& v) noexcept -> Eigen::Map<const Eigen::Vector3f> {
		return Eigen::Map<const Eigen::Vector3f>{v.v};
	}

	inline auto from_eigen(const Eigen::Vector3f& v) noexcept -> Vec3 {
		return Vec3{v.x(), v.y(), v.z()};
	}

	inline auto app_v(const Sophus::SE3f& h, const Vec3& v) noexcept -> Vec3 {
		return from_eigen(h * to_eigen(v));
	}

	inline auto app_uv(const Sophus::SE3f& h, const UVec3& uv) noexcept -> UVec3 {
		return UVec3{from_eigen(h.so3() * to_eigen(uv))};
	}

	inline auto trans(const Vec3& v) noexcept -> Sophus::SE3f {
		return Sophus::SE3f{Sophus::SO3f{}, to_eigen(v)};
	}

	inline auto rot(const Sophus::SO3f& r) noexcept -> Sophus::SE3f {
		return Sophus::SE3f{r, Eigen::Vector3f::Zero()};
	}

	// Yaw -> Pitch -> Roll の順に適用
	inline auto ypr(const Vec3& ypr) noexcept -> Sophus::SO3f {
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

		return Sophus::SO3f{Eigen::Quaternionf{
			cr * cp * cy - sr * sp * sy,
			sr * cp * cy + cr * sp * sy,
			cr * sp * cy - sr * cp * sy,
			cr * cp * sy + sr * sp * cy
		}};
	}

	// Roll -> Pitch -> Yaw の順に適用
	inline auto rpy(const Vec3& rpy) noexcept -> Sophus::SO3f {
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

		return Sophus::SO3f{Eigen::Quaternionf{
			cr * cp * cy + sr * sp * sy,
			sr * cp * cy - cr * sp * sy,
			cr * sp * cy + sr * cp * sy,
			cr * cp * sy - sr * sp * cy
		}};
	}
} // namespace sotoba::math

namespace sotoba {
	template <>
	struct Repr<Sophus::SE3f> final {
		static auto repr(const Sophus::SE3f& self) -> std::string {
			const auto& q = self.unit_quaternion().coeffs();
			return std::format(
				"SE3f{{Quaternion{{{}, {}, {}, {}}}, {}}}",
				q.x(),
				q.y(),
				q.z(),
				q.w(),
				Repr<math::Vec3>::repr(math::from_eigen(self.translation()))
			);
		}
	};

	template <>
	struct Repr<Sophus::SO3f> final {
		static auto repr(const Sophus::SO3f& self) -> std::string {
			const auto& q = self.unit_quaternion().coeffs();
			return std::format("SO3f{{Quaternion{{{}, {}, {}, {}}}}}", q.x(), q.y(), q.z(), q.w());
		}
	};
} // namespace sotoba

#ifdef sotoba_ENABLE_TESTING
	#include <cmath>
	#include <numbers>

	#include <doctest.h>

namespace sotoba::math {
	// 不変条件: q と -q は同じ回転なので、符号を揃えてから成分を比較する
	template <>
	struct ApproxCheckImpl<Sophus::SO3f> final {
		static auto
		compare(const Sophus::SO3f& l, const Sophus::SO3f& r, const std::optional<float> eps)
			-> bool {
			const Eigen::Vector4f lc = l.unit_quaternion().coeffs();
			Eigen::Vector4f rc = r.unit_quaternion().coeffs();
			if (lc.dot(rc) < 0.f) rc = -rc;
			return ApproxCheckImpl<Vec4>::compare(
				Vec4{lc[0], lc[1], lc[2], lc[3]},
				Vec4{rc[0], rc[1], rc[2], rc[3]},
				eps
			);
		}
	};

	template <>
	struct ApproxCheckImpl<Sophus::SE3f> final {
		static auto
		compare(const Sophus::SE3f& l, const Sophus::SE3f& r, const std::optional<float> eps)
			-> bool {
			return ApproxCheckImpl<Sophus::SO3f>::compare(l.so3(), r.so3(), eps)
				&& ApproxCheckImpl<Vec3>::compare(
					   from_eigen(l.translation()),
					   from_eigen(r.translation()),
					   eps
				   );
		}
	};
} // namespace sotoba::math

TEST_SUITE("se3.hpp") {
	using namespace sotoba::math;
	using namespace doctest;

	// テスト用定数: 円周率
	using std::numbers::pi;
	static constexpr float pi_half = pi / 2.0f;

	TEST_CASE("Identity (ide)") {
		auto I = Sophus::SE3f{};

		// 恒等変換: 回転なし(w=1), 並進なし(0,0,0)
		CHECK(I.translation().x() == 0.0f);
		CHECK(I.translation().y() == 0.0f);
		CHECK(I.translation().z() == 0.0f);

		// 実部は1.0
		CHECK(I.unit_quaternion().w() == Approx(1.0f));

		Vec3 v{1.0f, 2.0f, 3.0f};
		CHECK(ApproxCheck{app_v(I, v)} == ApproxCheck{v});
	}

	TEST_CASE("Apply Vector (app_v) with RPY Rotation") {
		// シナリオ:
		// 回転: Z軸周り(Yaw)に90度
		// 並進: X軸方向に +10
		// rpy(roll, pitch, yaw) -> rpy(0, 0, 90deg)

		const auto q_yaw_90 = ypr(Vec3{0.0f, 0.0f, pi_half});
		Sophus::SE3f T{q_yaw_90, Eigen::Vector3f{10.0f, 0.0f, 0.0f}};

		// 入力点: (1, 0, 0)
		Vec3 p_in{1.0f, 0.0f, 0.0f};

		// 計算過程:
		// 1. 回転: (1, 0, 0) -> Z軸90度回転 -> (0, 1, 0)
		// 2. 並進: (0, 1, 0) + (10, 0, 0) -> (10, 1, 0)
		Vec3 p_out = app_v(T, p_in);

		CHECK(p_out.x() == Approx(10.0f));
		CHECK(p_out.y() == Approx(1.0f));
		CHECK(p_out.z() == Approx(0.0f));
	}

	TEST_CASE("Apply Unit Vector (app_uv)") {
		// app_uv は「方向」の変換なので、SE3の並進成分を無視するはず

		const auto q_pitch_90 = ypr(Vec3{0.0f, pi_half, 0.0f}); // Y軸回転
		Sophus::SE3f T{q_pitch_90, Eigen::Vector3f{100.0f, 50.0f, -20.0f}}; // 大きな並進を設定

		// 入力: X軸方向 (1, 0, 0)
		UVec3 uv_in = vec::as_uvec(Vec3{1.0f, 0.0f, 0.0f}); // ※vecの仕様に合わせて生成

		// 期待値: X軸をY軸周りに90度回転 -> Z軸の負の方向
		UVec3 uv_out = app_uv(T, uv_in);

		// 並進成分の影響を受けていないこと (長さが1であること)
		CHECK(vec::fast_length(uv_out) == Approx(1.0f));

		// 回転の確認: (1,0,0) -> Y軸90度 -> (-sin, 0, cos)
		CHECK(uv_out.z() == Approx(-1.0f));
		CHECK(uv_out.x() == Approx(0.0f));
	}

	TEST_CASE("Inverse Transformation (inv)") {
		// 適当な複合変換
		const auto q = ypr(Vec3{0.1f, 0.2f, 0.3f});
		const Vec3 p{5.0f, -2.0f, 3.0f};
		Sophus::SE3f T{q, to_eigen(p)};

		// T * T_inv = Identity
		Sophus::SE3f T_inv = T.inverse();
		Sophus::SE3f res = T * T_inv;

		CHECK(ApproxCheck{res} == ApproxCheck{Sophus::SE3f{}});
	}

	TEST_CASE("Operator Multiplication (*)") {
		// T1: X移動 (+10)
		Sophus::SE3f T1 = trans(Vec3{10.0f, 0.0f, 0.0f});

		// T2: Z回転 (90度)
		Sophus::SE3f T2 = rot(ypr(Vec3{0.0f, 0.0f, pi_half}));

		// Case A: T1 * T2
		// 数式: (q1, p1) * (q2, p2) = (q1*q2, p1 + q1*p2)
		// 結果: 原点を中心に回転し、その座標系原点が(10,0,0)にある状態
		Sophus::SE3f T1_T2 = T1 * T2;

		Vec3 v{1.0f, 0.0f, 0.0f};
		// v を T1_T2 で変換:
		// Rot(90) * v -> (0, 1, 0)
		// + (10, 0, 0) -> (10, 1, 0)
		CHECK(ApproxCheck{app_v(T1_T2, v)} == ApproxCheck{Vec3{10.0f, 1.0f, 0.0f}});

		// Case B: T2 * T1 (順序逆)
		// p = 0 + RotZ(90) * (10, 0, 0) = (0, 10, 0)
		Sophus::SE3f T2_T1 = T2 * T1;

		// v を T2_T1 で変換:
		// Rot(90) * v -> (0, 1, 0)
		// + (0, 10, 0) -> (0, 11, 0)
		CHECK(ApproxCheck{app_v(T2_T1, v)} == ApproxCheck{Vec3{0.0f, 11.0f, 0.0f}});
	}

	TEST_CASE("Chaining methods (rot, trans)") {
		Sophus::SE3f base{};
		auto q = ypr(Vec3{0.0f, 0.0f, pi_half});
		Vec3 v{5.0f, 0.0f, 0.0f};

		// rot(q) -> SE3(q, 0) * base -> 純粋回転
		Sophus::SE3f T = trans(v) * rot(q) * base;

		// 期待される動作:
		// 1. まず回転 (z90)
		// 2. その後、ワールドX軸に+5

		Vec3 p{1.0f, 0.0f, 0.0f};
		// Rot(z90) * p -> (0, 1, 0)
		// Trans(5,0,0) + result -> (5, 1, 0)

		CHECK(ApproxCheck{app_v(T, p)} == ApproxCheck{Vec3{5.0f, 1.0f, 0.0f}});
	}

	TEST_CASE("Normalization") {
		auto q_bad = ypr(Vec3{0.0f, 0.0f, 0.1f});

		Sophus::SE3f T{q_bad, Eigen::Vector3f{1.0f, 1.0f, 1.0f}};
		Sophus::SE3f T_norm = T;
		T_norm.so3().normalize();

		CHECK(ApproxCheck{T} == ApproxCheck{T_norm});

		CHECK(T_norm.unit_quaternion().norm() == Approx(1.0f));
	}

	TEST_CASE("Construction and Identity") {
		auto q_id = Sophus::SO3f{};
		CHECK(q_id.unit_quaternion().x() == 0.f);
		CHECK(q_id.unit_quaternion().y() == 0.f);
		CHECK(q_id.unit_quaternion().z() == 0.f);
		CHECK(q_id.unit_quaternion().w() == 1.f);

		CHECK(q_id.unit_quaternion().norm() == Approx(1.f));
	}

	TEST_CASE("Conjugate") {
		const Eigen::Quaternionf q{4.f, 1.f, 2.f, 3.f};
		const Sophus::SO3f r{q};
		const auto r_conj = r.inverse();

		CHECK(r_conj.unit_quaternion().x() == Approx(-r.unit_quaternion().x()));
		CHECK(r_conj.unit_quaternion().y() == Approx(-r.unit_quaternion().y()));
		CHECK(r_conj.unit_quaternion().z() == Approx(-r.unit_quaternion().z()));
		CHECK(r_conj.unit_quaternion().w() == Approx(r.unit_quaternion().w()));
	}

	TEST_CASE("Multiplication (Hamilton Product)") {
		const Sophus::SO3f qi{Eigen::Quaternionf{0.f, 1.f, 0.f, 0.f}};
		const Sophus::SO3f qj{Eigen::Quaternionf{0.f, 0.f, 1.f, 0.f}};
		const Sophus::SO3f qk{Eigen::Quaternionf{0.f, 0.f, 0.f, 1.f}};

		auto res_k = qi * qj;
		CHECK(ApproxCheck{res_k} == ApproxCheck{qk});

		auto res_neg_k = qj * qi;
		const Sophus::SO3f neg_qk{Eigen::Quaternionf{0.f, 0.f, 0.f, -1.f}};
		CHECK(ApproxCheck{res_neg_k} == ApproxCheck{neg_qk});

		auto res_neg_1 = qi * qi;
		const Sophus::SO3f neg_one{Eigen::Quaternionf{-1.f, 0.f, 0.f, 0.f}};
		CHECK(ApproxCheck{res_neg_1} == ApproxCheck{neg_one});

		const Sophus::SO3f q{Eigen::Quaternionf{4.f, 1.f, 2.f, 3.f}};
		CHECK(ApproxCheck{q * Sophus::SO3f{}} == ApproxCheck{q});
		CHECK(ApproxCheck{Sophus::SO3f{} * q} == ApproxCheck{q});
	}

	TEST_CASE("Quaternion normalization") {
		const Sophus::SO3f uq{Eigen::Quaternionf{1.f, 1.f, 1.f, 1.f}};

		CHECK(uq.unit_quaternion().norm() == Approx(1.f));

		CHECK(uq.unit_quaternion().x() == Approx(0.5f));
		CHECK(uq.unit_quaternion().w() == Approx(0.5f));
	}

	TEST_CASE("RPY (Euler Angles) Construction") {
		constexpr float epsilon = 1e-5f;

		auto q_roll = ypr(Vec3{pi_half, 0.f, 0.f});
		CHECK(q_roll.unit_quaternion().x() == Approx(std::sin(pi / 4.f)));
		CHECK(q_roll.unit_quaternion().w() == Approx(std::cos(pi / 4.f)));
		CHECK(q_roll.unit_quaternion().y() == Approx(0.f).epsilon(epsilon));
		CHECK(q_roll.unit_quaternion().z() == Approx(0.f).epsilon(epsilon));

		auto q_yaw = ypr(Vec3{0.f, 0.f, pi_half});
		CHECK(q_yaw.unit_quaternion().z() == Approx(std::sin(pi / 4.f)));
		CHECK(q_yaw.unit_quaternion().w() == Approx(std::cos(pi / 4.f)));
	}

	TEST_CASE("Vector Rotation (rot)") {
		constexpr float epsilon = 1e-5f;
		Vec3 v_in{1.f, 0.f, 0.f};

		SUBCASE("Rotate +90 deg around Z-axis (Yaw)") {
			auto q = ypr(Vec3{0.f, 0.f, pi_half});
			Vec3 v_out = app_v(rot(q), v_in);

			CHECK(v_out.x() == Approx(0.f).epsilon(epsilon));
			CHECK(v_out.y() == Approx(1.f).epsilon(epsilon));
			CHECK(v_out.z() == Approx(0.f).epsilon(epsilon));
		}

		SUBCASE("Rotate +90 deg around Y-axis (Pitch)") {
			auto q = ypr(Vec3{0.f, pi_half, 0.f});
			Vec3 v_out = app_v(rot(q), v_in);

			CHECK(v_out.x() == Approx(0.f).epsilon(epsilon));
			CHECK(v_out.y() == Approx(0.f).epsilon(epsilon));
			CHECK(v_out.z() == Approx(-1.f).epsilon(epsilon));
		}

		SUBCASE("Identity rotation") {
			Vec3 v_out = app_v(rot(Sophus::SO3f{}), v_in);
			CHECK(ApproxCheck{v_out} == ApproxCheck{v_in});
		}
	}

	TEST_CASE("Unit Vector Rotation (app_uv)") {
		constexpr float epsilon = 1e-5f;
		UVec3 v_in{1.f, 0.f, 0.f};
		auto q = ypr(Vec3{0.f, 0.f, pi_half});

		UVec3 v_out = app_uv(rot(q), v_in);

		CHECK(v_out.x() == Approx(0.f).epsilon(epsilon));
		CHECK(v_out.y() == Approx(1.f).epsilon(epsilon));

		static_assert(std::is_same_v<decltype(v_out), UVec3>);
	}

	TEST_CASE("rpy: roll -> pitch -> yaw の順に適用される") {
		const Vec3 angles{0.3f, -0.2f, 0.5f};
		const Eigen::Matrix3f expected =
			(Eigen::AngleAxisf{angles.z(), Eigen::Vector3f::UnitZ()}
			 * Eigen::AngleAxisf{angles.y(), Eigen::Vector3f::UnitY()}
			 * Eigen::AngleAxisf{angles.x(), Eigen::Vector3f::UnitX()})
				.toRotationMatrix();

		CHECK(rpy(angles).matrix().isApprox(expected, 1e-5f));
	}

	TEST_CASE("ypr: yaw -> pitch -> roll の順に適用される") {
		const Vec3 angles{0.3f, -0.2f, 0.5f};
		const Eigen::Matrix3f expected =
			(Eigen::AngleAxisf{angles.x(), Eigen::Vector3f::UnitX()}
			 * Eigen::AngleAxisf{angles.y(), Eigen::Vector3f::UnitY()}
			 * Eigen::AngleAxisf{angles.z(), Eigen::Vector3f::UnitZ()})
				.toRotationMatrix();

		CHECK(ypr(angles).matrix().isApprox(expected, 1e-5f));
	}

	TEST_CASE("to_eigen/from_eigen: ゼロコピーの往復が恒等") {
		const Vec3 v{1.f, -2.f, 3.f};
		CHECK(to_eigen(v).data() == v.v);
		CHECK(ApproxCheck{from_eigen(to_eigen(v))} == ApproxCheck{v});
	}
}

#endif
