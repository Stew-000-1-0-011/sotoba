#pragma once

#include <algorithm>
#include <limits>
#include "sotoba/math/epsilon.hpp"
#include "sotoba/math/quaternion.hpp"
#include "sotoba/math/scalar_functions.hpp"
#include "sotoba/math/se3.hpp"
#include "sotoba/math/square_mat.hpp"
#include "sotoba/math/vec.hpp"
#include "sotoba/stdtypes.hpp"
#include "sotoba/surface/surface.hpp"

namespace sotoba::surface::box_impl {
	using math::SE3;
	using math::SquareMat;
	using math::UVec3;
	using math::Vec;
	using math::Vec3;
	using math::Vec4;
	namespace vec = math::vec;

	struct BoxOuter final {
		Vec3 center;
		u32 wall_exist;
		SquareMat<3> rot;
		Vec3 hlens;

		BoxOuter(
			const Vec3& center,
			const SquareMat<3>& rot,
			const Vec3& hlens,
			const std::array<bool, 6> wall_not_exist = {}
		) noexcept
			: center{center}, wall_exist{}, rot{rot}, hlens{hlens} {
			for (u8 i = 0; i < 6; ++i) wall_exist |= u32(wall_not_exist[i] ? 0 : 1) << i;
		}

		BoxOuter(const BoxOuter&) noexcept = default;
		BoxOuter(BoxOuter&&) noexcept = default;
		auto operator=(const BoxOuter&) -> BoxOuter& = default;
		auto operator=(BoxOuter&&) -> BoxOuter& = default;

		auto closest_pd(const Vec3& p) const noexcept -> Vec4 {
			const Vec3 co_local = this->rot * -this->center;
			const Vec3 cp_local = this->rot * (p - this->center);
			Vec3 clamped_cp_local;
			for (u8 i = 0; i < 3; ++i) {
				clamped_cp_local[i] = math::clamp(cp_local[i], -this->hlens[i], this->hlens[i]);
			}

			Vec3 cq_local{};
			float d = std::numeric_limits<float>::infinity();
			for (u8 i = 0; i < 3; ++i) {
				if ((wall_exist & u32(1) << 2 * i) && co_local[i] < -this->hlens[i]) {
					Vec3 cq_local_ = clamped_cp_local;
					cq_local_[i] = -this->hlens[i];
					if (const float d_ = vec::distance2(cq_local_, cp_local); d_ < d) {
						d = d_;
						cq_local = cq_local_;
					}
				} else if ((wall_exist & u32(1) << (2 * i + 1)) && this->hlens[i] < co_local[i]) {
					Vec3 cq_local_ = clamped_cp_local;
					cq_local_[i] = this->hlens[i];
					if (const float d_ = vec::distance2(cq_local_, cp_local); d_ < d) {
						d = d_;
						cq_local = cq_local_;
					}
				}
			}

			return {this->center + this->rot.transpose() * cq_local, Vec{d}};
		}

		auto closest_pdn(const Vec3& p) const noexcept -> std::pair<Vec4, UVec3> {
			const Vec3 co_local = this->rot * -this->center;
			const Vec3 cp_local = this->rot * (p - this->center);
			Vec3 clamped_cp_local;
			for (u8 i = 0; i < 3; ++i) {
				clamped_cp_local[i] = math::clamp(cp_local[i], -this->hlens[i], this->hlens[i]);
			}

			Vec3 cq_local{};
			UVec3 n{};
			float d = std::numeric_limits<float>::infinity();
			for (u8 i = 0; i < 3; ++i) {
				if ((wall_exist & u32(1) << 2 * i) && co_local[i] < -this->hlens[i]) {
					Vec3 cq_local_ = clamped_cp_local;
					cq_local_[i] = -this->hlens[i];
					if (const float d_ = vec::distance2(cq_local_, cp_local); d_ < d) {
						d = d_;
						n = -this->rot[i];
						cq_local = cq_local_;
					}
				} else if ((wall_exist & u32(1) << (2 * i + 1)) && this->hlens[i] < co_local[i]) {
					Vec3 cq_local_ = clamped_cp_local;
					cq_local_[i] = this->hlens[i];
					if (const float d_ = vec::distance2(cq_local_, cp_local); d_ < d) {
						d = d_;
						n = this->rot[i];
						cq_local = cq_local_;
					}
				}
			}

			return {{this->center + this->rot.transpose() * cq_local, Vec{d}}, n};
		}

		void apply_se3(const SE3& h) noexcept {
			this->center = h.app_v(this->center);
			for (u8 i = 0; i < 3; ++i) {
				const UVec3 new_axis = h.app_uv(vec::as_uvec(this->rot[i]));
				for (u8 j = 0; j < 3; ++j) this->rot[i, j] = new_axis[j];
			}
		}

		auto ray_collision(const UVec3& ray) const noexcept -> float {
			// Slabs法
			const Vec3 co_local = this->rot * -this->center;
			float t_enter = -std::numeric_limits<float>::infinity();
			float t_exit = std::numeric_limits<float>::infinity();
			for (u8 i = 0; i < 3; ++i) {
				const Vec3 axis = this->rot[i];
				const float ray_d = vec::dot(ray, axis);
				if (math::fabs(ray_d) < math::epsilon) {
					if (!(-hlens[i] < co_local[i] && co_local[i] < hlens[i]))
						return std::numeric_limits<float>::infinity();
					else continue;
				}

				const float t1 = (-this->hlens[i] - co_local[i]) / ray_d;
				const float t2 = (this->hlens[i] - co_local[i]) / ray_d;
				const auto [tmin, tmax] = std::minmax(t1, t2);
				t_enter = std::max(t_enter, tmin);
				t_exit = std::min(t_exit, tmax);
			}

			if (t_enter <= 0 || t_exit < t_enter) { return std::numeric_limits<float>::infinity(); }

			return math::pow2(t_enter);
		}
	};

	static_assert(surfacelike<BoxOuter>);
} // namespace sotoba::surface::box_impl

namespace sotoba::surface {
	using box_impl::BoxOuter;
}

#ifdef sotoba_ENABLE_TESTING

	#include <doctest.h>

TEST_SUITE("box.hpp") {
	using sotoba::surface::box_impl::BoxOuter;
	using namespace sotoba::math;

	TEST_CASE("BoxOuter") {
		// 基本設定: 中心(0,0,5), 半サイズ(1,1,1), 回転なし
		// 原点(LiDAR)からは Box の -Z 面（ワールド座標 z=4）が見えるはず
		const Vec3 center{0.0f, 0.0f, 5.0f};
		const SquareMat<3> rot = SquareMat<3>::ide();
		const Vec3 hlens{1.0f, 1.0f, 1.0f};
		BoxOuter box{center, rot, hlens};

		SUBCASE("closest_pd: 可視面への正対") {
			// 原点とボックスの間にある点
			const Vec3 p{0.0f, 0.0f, 0.0f};
			const auto res = box.closest_pd(p);

			// 最接近点は z=4 の面上の点 (0, 0, 4)
			// 距離の二乗は (0,0,4) - (0,0,0) => 16
			const Vec4 expected{0.0f, 0.0f, 4.0f, 16.0f};
			CHECK(ApproxCheck{res} == ApproxCheck{expected});
		}

		SUBCASE("closest_pd: 可視面からはみ出た点（クランプ動作）") {
			// x方向にずれた点 (2, 0, 4)
			// ボックスのx範囲は [-1, 1] なので、可視面(z=4)上でクランプされて (1, 0, 4) になるはず
			const Vec3 p{2.0f, 0.0f, 4.0f};
			const auto res = box.closest_pd(p);

			// 距離: p(2,0,4) - q(1,0,4) => dist 1 => sq 1
			const Vec4 expected{1.0f, 0.0f, 4.0f, 1.0f};
			CHECK(ApproxCheck{res} == ApproxCheck{expected});
		}

		SUBCASE("closest_pd: 幾何学的判定のコーナーケース") {
			// 修正されたバグの回帰テスト
			// 状況: 原点から見て、左面(-X)と下面(-Y)が見える位置にボックスを置く
			// Box中心: (2, 2, 0), hlens: (1, 1, 1) -> 範囲 x:[1,3], y:[1,3]
			BoxOuter corner_box{Vec3{2.0f, 2.0f, 0.0f}, rot, hlens};

			// テスト点P: (0.8, 0.9, 0)
			// 左面 (x=1) までの距離: |0.8 - 1.0| = 0.2.  y=0.9は範囲外(1~3) -> クランプでy=1.0.
			//    -> 左面上の点(1.0, 1.0, 0). Pからの距離^2 = 0.2^2 + 0.1^2 = 0.04 + 0.01 = 0.05
			// 下面 (y=1) までの距離: |0.9 - 1.0| = 0.1.  x=0.8は範囲外(1~3) -> クランプでx=1.0.
			//    -> 下面上の点(1.0, 1.0, 0). Pからの距離^2 = 0.2^2 + 0.1^2 = 0.05

			// より明確な差が出るケース:
			// P = (0.8, -10.0, 0) -> 明らかに左面(x=1)の方が近い。下面(y=1)はずっと遠い。
			// 以前のバグ(1次元距離比較)だと:
			//   x面への垂直距離: |0.8 - 1| = 0.2
			//   y面への垂直距離: |-10 - 1| = 11.0
			//   これは1次元でも成立する。

			// 1次元距離だと誤判定する微妙なケース:
			// Box: x[-1, 1], y[-1, 1]. 原点(-10, -10, 0).
			// 左面(x=-1)と下面(y=-1)が見える。
			BoxOuter origin_box{Vec3{0.0f, 0.0f, 0.0f}, rot, hlens};
			// P = (-1.2, -0.9, 0).
			// 左面(x=-1)への垂直距離: |-1.2 - (-1)| = 0.2.
			//    y=-0.9は範囲内なのでクランプ移動なし。3D距離も 0.2。
			// 下面(y=-1)への垂直距離: |-0.9 - (-1)| = 0.1.
			//    x=-1.2は範囲外なので x=-1にクランプ。
			//    3D距離は sqrt( (-1.2 - -1)^2 + (-0.9 - -1)^2 ) = sqrt(0.04 + 0.01) = sqrt(0.05) = 0.223...
			// 正解: 左面 (距離0.2)。
			// 誤り(1次元のみ): 下面 (距離0.1)。

			// 原点を移動させるのではなく、Boxを相対的に配置して再現
			// Boxを (2, 2, 0) に配置。原点は(0,0,0)。
			// 見える面: -X (x=1), -Y (y=1).
			// P = (0.8, 1.1, 0).
			// -X面(x=1): 垂直diff 0.2. y=1.1は範囲内[1,3]. -> 3D距離 0.2.
			// -Y面(y=1): 垂直diff 0.1. x=0.8は範囲外[1,3]. -> x=1にクランプ.
			//    -> 3D距離 sqrt((0.8-1)^2 + (1.1-1)^2) = sqrt(0.04 + 0.01) = 0.223.
			// よって -X面 が選ばれるべき。

			const Vec3 p_trick{0.8f, 1.1f, 0.0f};
			const auto res = corner_box.closest_pd(p_trick);

			// 正解: -X面上の点 (1.0, 1.1, 0.0)
			const Vec4 expected_pos{1.0f, 1.1f, 0.0f, 0.04f}; // 0.2^2
			CHECK(ApproxCheck{res} == ApproxCheck{expected_pos});

			// 法線付き版もチェック (-X面なので法線は -1, 0, 0)
			const auto [res_n, norm] = corner_box.closest_pdn(p_trick);
			CHECK(ApproxCheck{res_n} == ApproxCheck{expected_pos});
			CHECK(ApproxCheck{Vec3(norm)} == ApproxCheck{Vec3{-1.0f, 0.0f, 0.0f}});
		}

		SUBCASE("closest_pd: 原点がボックス内部にある場合") {
			// 原点(0,0,0)を含んでいるボックス
			BoxOuter inside_box{Vec3{0.0f, 0.0f, 0.0f}, rot, hlens};

			const auto res = inside_box.closest_pd(Vec3{10.0f, 0.0f, 0.0f});

			// 仕様により infinity を返す
			CHECK(res.w() == std::numeric_limits<float>::infinity());
		}

		SUBCASE("apply_se3: 座標変換") {
			// +X方向に2移動し、Z軸回りに90度回転させる変換
			// 元のBox: 中心(0,0,5), 半径(1,1,1)
			// 移動後中心: (2, 0, 5) -> 回転(Z90) -> (0, 2, 5) ... 注意: app_vの仕様によるが、通常は Rot * v + trans

			const UQuaternion rot_z90 = quaternion::ypr({0.f, 0.f, std::numbers::pi_v<float> / 2});
			const Vec3 trans{2.0f, 0.0f, 0.0f};
			SE3 pose{rot_z90, trans};

			box.apply_se3(pose);

			// 中心位置の確認: (0,0,5) -> rot -> (0,0,5) -> + trans(2,0,0) -> (2,0,5)
			// ※ ライブラリの app_v が (R*v + t) か (v + t) か確認必要だが、一般的にRigidBody変換ならこう。
			// テストコード上では計算結果をcheck_vec_approxする
			const Vec3 expected_center = pose.app_v(Vec3{0.0f, 0.0f, 5.0f});
			CHECK(ApproxCheck{box.center} == ApproxCheck{expected_center});

			// 回転の確認: X軸(1,0,0)だったものがY軸(0,1,0)になっているか
			// box.rot は行ベクトルか列ベクトルかでアクセスが変わるが、apply_se3の実装を見る限り列更新
			// rot[0] (X軸) は (0, 1, 0) になっているはず
			const Vec3 new_axis_x = box.rot[0];
			CHECK(ApproxCheck{new_axis_x} == ApproxCheck{Vec3{0.0f, 1.0f, 0.0f}});
		}

		SUBCASE("ray_collision: 衝突判定") {
			// Box: (0, 0, 5), size 1
			// Ray: (0, 0, 1) -> 正面衝突
			const UVec3 ray_hit{0.0f, 0.0f, 1.0f};
			const auto hit_res = box.ray_collision(ray_hit);

			// 衝突点は (0, 0, 4), 距離4 (t=4)
			CHECK(hit_res == 16.f);
		}

		SUBCASE("ray_collision: 平行レイの処理") {
			// Box: (0, 0, 5), size 1 => x範囲 [-1, 1]
			// Ray: (1, 0, 0) -> x軸平行
			// Boxはz=5にあるので、z=0発射のxレイは絶対に当たらない
			const UVec3 ray_parallel{1.0f, 0.0f, 0.0f};
			const auto miss_res = box.ray_collision(ray_parallel);

			CHECK(miss_res == std::numeric_limits<float>::infinity());
		}

		SUBCASE("ray_collision: 内部からの平行レイ（原点が内部にある場合）") {
			// Boxを原点に移動
			BoxOuter zero_box{Vec3{0.0f, 0.0f, 0.0f}, rot, hlens};
			// x軸平行レイ。幾何学的には貫通しているが、
			// 仕様「原点<=0 || exit < enter」により非衝突(inf)になるはず
			const UVec3 ray_parallel{1.0f, 0.0f, 0.0f};
			const auto res = zero_box.ray_collision(ray_parallel);

			CHECK(res == std::numeric_limits<float>::infinity());
		}
	}
}
#endif