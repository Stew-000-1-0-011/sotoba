#pragma once

#include <format>
#include "sotoba/math/approx_check.hpp"
#include "sotoba/math/vec_forward_decl.hpp"
#include "sotoba/repr.hpp"

#include "quaternion.hpp"
#include "vec.hpp"

namespace sotoba::math {
	// 剛体変換
	struct SE3 final {
		UQuaternion uq{math::quaternion::ide()};
		Vec3 p{};

		static constexpr auto ide() noexcept -> SE3 {
			return SE3{quaternion::ide(), Vec3{}};
		}

		static constexpr auto rot(const UQuaternion& uq) noexcept -> SE3 {
			return SE3{uq, Vec3{}};
		}

		static constexpr auto trans(const Vec3& v) noexcept -> SE3 {
			return SE3{quaternion::ide(), v};
		}

		constexpr auto app_v(const Vec3& v) const noexcept -> Vec3 {
			return rot_vec(this->uq, v) + this->p;
		}

		constexpr auto app_uv(const UVec3& uv) const noexcept -> UVec3 {
			return rot_uvec(this->uq, uv);
		}

		constexpr auto normalize() noexcept -> SE3 {
			return SE3{this->uq.normalize(), p};
		}

		constexpr auto inv() const noexcept -> SE3 {
			const auto conj_q = this->uq.conj();
			return SE3{conj_q, -rot_vec(conj_q, this->p)};
		}

		constexpr friend auto operator*(const SE3& l, const SE3& r) -> SE3 {
			return SE3{l.uq * r.uq, l.p + rot_vec(l.uq, r.p)};
		}
	};
} // namespace sotoba::math

namespace sotoba {
	template <>
	struct Repr<math::SE3> final {
		static auto repr(const math::SE3& self) -> std::string {
			return std::format(
				"SE3{{{}, {}}}",
				Repr<math::UQuaternion>::repr(self.uq),
				Repr<math::Vec3>::repr(self.p)
			);
		}
	};
} // namespace sotoba

#ifdef sotoba_ENABLE_TESTING
	#include <cmath>

	#include <doctest.h>

namespace sotoba::math {
	template <>
	struct ApproxCheckImpl<SE3> final {
		static auto compare(const SE3& l, const SE3& r, const std::optional<float> eps) -> bool {
			return ApproxCheckImpl<UQuaternion>::compare(l.uq, r.uq, eps)
				&& ApproxCheckImpl<Vec3>::compare(l.p, r.p, eps);
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
		auto I = SE3::ide();

		// 恒等変換: 回転なし(w=1), 並進なし(0,0,0)
		CHECK(I.p.x() == 0.0f);
		CHECK(I.p.y() == 0.0f);
		CHECK(I.p.z() == 0.0f);

		// 実部は1.0 (ide()の実装に基づく)
		CHECK(I.uq.v.w() == Approx(1.0f));

		Vec3 v{1.0f, 2.0f, 3.0f};
		CHECK(ApproxCheck{I.app_v(v)} == ApproxCheck{v});
	}

	TEST_CASE("Apply Vector (app_v) with RPY Rotation") {
		// シナリオ:
		// 回転: Z軸周り(Yaw)に90度
		// 並進: X軸方向に +10
		// rpy(roll, pitch, yaw) -> rpy(0, 0, 90deg)

		const auto q_yaw_90 = quaternion::ypr(Vec3{0.0f, 0.0f, pi_half});
		SE3 T{q_yaw_90, Vec3{10.0f, 0.0f, 0.0f}};

		// 入力点: (1, 0, 0)
		Vec3 p_in{1.0f, 0.0f, 0.0f};

		// 計算過程:
		// 1. 回転: (1, 0, 0) -> Z軸90度回転 -> (0, 1, 0)
		// 2. 並進: (0, 1, 0) + (10, 0, 0) -> (10, 1, 0)
		Vec3 p_out = T.app_v(p_in);

		CHECK(p_out.x() == Approx(10.0f));
		CHECK(p_out.y() == Approx(1.0f));
		CHECK(p_out.z() == Approx(0.0f));
	}

	TEST_CASE("Apply Unit Vector (app_uv)") {
		// app_uv は「方向」の変換なので、SE3の並進成分(p)を無視するはず

		const auto q_pitch_90 = quaternion::ypr(Vec3{0.0f, pi_half, 0.0f}); // Y軸回転
		SE3 T{q_pitch_90, Vec3{100.0f, 50.0f, -20.0f}}; // 大きな並進を設定

		// 入力: X軸方向 (1, 0, 0)
		UVec3 uv_in = vec::as_uvec(Vec3{1.0f, 0.0f, 0.0f}); // ※vecの仕様に合わせて生成

		// 期待値: X軸をY軸周りに90度回転 -> Z軸の負の方向
		UVec3 uv_out = T.app_uv(uv_in);

		// 並進成分の影響を受けていないこと (長さが1であること)
		CHECK(vec::fast_length(uv_out) == Approx(1.0f));

		// 回転の確認: (1,0,0) -> Y軸90度 -> (-sin, 0, cos)
		CHECK(uv_out.z() == Approx(-1.0f));
		CHECK(uv_out.x() == Approx(0.0f));
	}

	TEST_CASE("Inverse Transformation (inv)") {
		// 適当な複合変換
		const auto q = quaternion::ypr(Vec3{0.1f, 0.2f, 0.3f});
		const Vec3 p{5.0f, -2.0f, 3.0f};
		SE3 T{q, p};

		// T * T_inv = Identity
		SE3 T_inv = T.inv();
		SE3 res = T * T_inv;

		CHECK(ApproxCheck{res} == ApproxCheck{SE3::ide()});
	}

	TEST_CASE("Operator Multiplication (*)") {
		// T1: X移動 (+10)
		SE3 T1{quaternion::ide(), Vec3{10.0f, 0.0f, 0.0f}};

		// T2: Z回転 (90度)
		SE3 T2{quaternion::ypr(Vec3{0.0f, 0.0f, pi_half}), Vec3{0.0f, 0.0f, 0.0f}};

		// Case A: T1 * T2
		// 数式: (q1, p1) * (q2, p2) = (q1*q2, p1 + q1*p2)
		// 結果: 原点を中心に回転し、その座標系原点が(10,0,0)にある状態
		SE3 T1_T2 = T1 * T2;

		Vec3 v{1.0f, 0.0f, 0.0f};
		// v を T1_T2 で変換:
		// Rot(90) * v -> (0, 1, 0)
		// + (10, 0, 0) -> (10, 1, 0)
		CHECK(ApproxCheck{T1_T2.app_v(v)} == ApproxCheck{Vec3{10.0f, 1.0f, 0.0f}});

		// Case B: T2 * T1 (順序逆)
		// p = 0 + RotZ(90) * (10, 0, 0) = (0, 10, 0)
		SE3 T2_T1 = T2 * T1;

		// v を T2_T1 で変換:
		// Rot(90) * v -> (0, 1, 0)
		// + (0, 10, 0) -> (0, 11, 0)
		CHECK(ApproxCheck{T2_T1.app_v(v)} == ApproxCheck{Vec3{0.0f, 11.0f, 0.0f}});
	}

	TEST_CASE("Chaining methods (rot, trans)") {
		SE3 base = SE3::ide();
		auto q = quaternion::ypr(Vec3{0.0f, 0.0f, pi_half});
		Vec3 v{5.0f, 0.0f, 0.0f};

		// base.rot(q) -> SE3(q, 0) * base -> 純粋回転
		SE3 T = SE3::trans(v) * base.rot(q);

		// 期待される動作:
		// 1. まず回転 (z90)
		// 2. その後、ワールドX軸に+5

		Vec3 p{1.0f, 0.0f, 0.0f};
		// Rot(z90) * p -> (0, 1, 0)
		// Trans(5,0,0) + result -> (5, 1, 0)

		CHECK(ApproxCheck{T.app_v(p)} == ApproxCheck{Vec3{5.0f, 1.0f, 0.0f}});
	}

	TEST_CASE("Normalization") {
		// 演算誤差を模擬して、少しずれたSE3を作る
		auto q_bad = quaternion::ypr(Vec3{0.0f, 0.0f, 0.1f});
		// 強引にw成分を書き換えて非正規化したいが、privateメンバなので
		// ここでは normalize() を呼んでクラッシュせず、値が維持されるかを確認

		SE3 T{q_bad, Vec3{1.0f, 1.0f, 1.0f}};
		SE3 T_norm = T.normalize();

		CHECK(ApproxCheck{T} == ApproxCheck{T_norm});

		// 返り値のuqがちゃんと正規化されているか
		CHECK(T_norm.uq.fast_length() == Approx(1.0f));
	}
}

#endif
