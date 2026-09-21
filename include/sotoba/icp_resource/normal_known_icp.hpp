#pragma once

#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/src/Cholesky/LLT.h>
#include <Eigen/src/Core/util/Constants.h>

#include "sotoba/math/sym_mat.hpp"
#include "sotoba/math/vec_forward_decl.hpp"
#include "sotoba/stdtypes.hpp"
#include "sotoba/surf_obj_id.hpp"

#include "sotoba/math/quaternion.hpp"
#include "sotoba/math/se3.hpp"
#include "sotoba/math/square_mat.hpp"
#include "sotoba/math/vec.hpp"
#include "sotoba/surface/surface.hpp"

#include "resource.hpp"

namespace sotoba::icp_resource::normal_known_icp_impl {
	using math::SE3;
	using math::SquareMat;
	using math::SymMat;
	using math::UVec3;
	using math::Vec;
	using math::Vec3;
	using math::Vec4;
	using Vec6 = Vec<6>;
	namespace vec = math::vec;
	using surface::ExplanationOnlySurface;
	using surface::surfacelike;
	using icp_resource::IcpError;
	using icp_resource::ObjStatus;

	template <surfacelike... Surfaces_>
	struct NormalKnownResource final {
		// 表面とその情報、座標変換後の表面のバッファ
		std::tuple<std::vector<Surfaces_>...> surfs;
		std::tuple<std::vector<Surfaces_>...> moved_surfs;
		std::array<std::vector<ObjSurfId>, sizeof...(Surfaces_)> osids;

		// 点群とその最近接点に関する情報
		// Vec4: [x,y,z,距離]
		std::vector<std::pair<std::pair<Vec4, UVec3>, ObjSurfId>> qs;

		// 加算されていくやつら
		std::vector<Vec6> b;
		std::vector<SymMat<3>> a_w;
		std::vector<SymMat<3>> a_t;
		std::vector<SquareMat<3>> a_wt;
		std::vector<usize> counts;

		// ここに入れた姿勢をもとに、ICPがはしり、補正された結果がここに入る
		std::vector<SE3> obj_poses;

		// 直近の run_icp におけるオブジェクトごとの姿勢更新結果
		std::vector<ObjStatus> obj_statuses;
		// 直近の run_icp で実際に回ったループ回数
		u32 loop_count;

		u8 obj_num;

		NormalKnownResource(
			std::tuple<std::vector<Surfaces_>...>&& surfs,
			std::array<std::vector<ObjSurfId>, sizeof...(Surfaces_)>&& osids,
			const u8 obj_num,
			const usize points_num
		) noexcept
			: surfs{std::move(surfs)}
			, moved_surfs{}
			, osids{std::move(osids)}
			, qs{}
			, b{}
			, a_w{}
			, a_t{}
			, a_wt{}
			, counts{}
			, obj_poses{}
			, obj_statuses{}
			, loop_count{0}
			, obj_num{obj_num} {
			(
				[&]<surfacelike S_>() {
					std::get<std::vector<S_>>(this->moved_surfs) =
						std::get<std::vector<S_>>(this->surfs);
				}.template operator()<Surfaces_>(),
				...
			);
			this->qs.resize(points_num);
			this->b.resize(obj_num);
			this->a_w.resize(obj_num);
			this->a_t.resize(obj_num);
			this->a_wt.resize(obj_num);
			this->counts.resize(obj_num);
			this->obj_poses.resize(obj_num, SE3::ide());
			this->obj_statuses.resize(obj_num, ObjStatus::not_run);
		}

		decltype(auto) obj_pose(this auto&& self, const u8 oid) noexcept {
			return self.obj_poses[oid];
		}

		// このバッファが受け入れられる最大点数
		auto points_capacity() const noexcept -> usize {
			return this->qs.size();
		}

		// 直近の run_icp におけるオブジェクトの姿勢更新結果
		auto obj_status(const u8 oid) const noexcept -> ObjStatus {
			return this->obj_statuses[oid];
		}

		// 直近の run_icp におけるオブジェクトの対応点数
		auto correspondence_count(const u8 oid) const noexcept -> usize {
			return this->counts[oid];
		}

		// 直近の run_icp で実際に回ったループ回数 (常に max_loop_num 以下)
		auto last_loop_count() const noexcept -> u32 {
			return this->loop_count;
		}

		// 点が少なすぎて姿勢を更新しない対応点数の閾値。変更しないこと。
		static constexpr usize min_correspondences = 3;

		/// 点対面ICPを走らせ、obj_poses を更新する。
		///
		/// point_cloud はセンサ座標系の点群。面の可視性判定はセンサ原点(0,0,0)を
		/// 基準に行うため、obj_pose には「マップ座標系の形状をセンサ座標系へ写す変換」
		/// (= 自己位置の逆変換) を入れること。向きを取り違えると全点が不可視になる。
		///
		/// accept_distance2 は対応点として受け入れる距離の **二乗**。
		///
		/// point_cloud.size() が points_capacity() を超える場合、バッファの再確保は
		/// 行わず IcpError::too_many_points を返す。このとき姿勢・状態は一切変化しない。
		///
		/// ループ回数は max_loop_num をハード上限とし、これを超えて回ることはない。
		/// 姿勢を更新した全オブジェクトの更新量 delta2 の最大値が convergence_delta2
		/// 以下になった時点で打ち切るため、実際の回数は常に max_loop_num 以下になる
		/// (last_loop_count() で取得できる)。
		/// 更新されたオブジェクトが1つも無い場合 (全て too_few_correspondences や
		/// solve_failed の場合) は最大値が 0 のままなので、既定の
		/// convergence_delta2 = 0.f でも1回で打ち切られる。姿勢が動かない以上
		/// 回し続けても結果は変わらないため、これは意図した挙動。
		auto run_icp(
			std::span<const Vec3> point_cloud,
			const Vec6& tikhonov,
			const u32 max_loop_num,
			const float accept_distance2,
			const float convergence_delta2 = 0.f
		) noexcept -> IcpError {
			if (point_cloud.size() > this->qs.size()) return IcpError::too_many_points;

			const auto tikhonov_w = vec::split<0, 3>(tikhonov);
			const auto tikhonov_t = vec::split<3, 6>(tikhonov);

			this->loop_count = 0;

			for (u32 iloop = 0; iloop < max_loop_num; ++iloop) {
				this->loop_count = iloop + 1;

				// surfsをobj_posesに従い移動
				[&]<usize... idxs_>(std::index_sequence<idxs_...>) {
					(
						[&]<surfacelike S_>(
							const std::vector<S_>& surf,
							const std::vector<ObjSurfId>& osid,
							std::vector<S_>& moved_surf
						) {
							for (usize i = 0; i < surf.size(); ++i) {
								const auto [oid, sid] = osid_depack(osid[i]);
								moved_surf[i] = surf[i];
								moved_surf[i].apply_se3(this->obj_poses[u8(oid)]);
							}
						}(std::get<idxs_>(this->surfs),
						  this->osids[idxs_],
						  std::get<idxs_>(this->moved_surfs)),
						...
					);
				}(std::index_sequence_for<Surfaces_...>{});

				// 各点の最近接点をqsに格納
				[&]<usize... idxs_>(std::index_sequence<idxs_...>) {
					for (usize ip = 0; ip < point_cloud.size(); ++ip) {
						if (!vec::isfinite(point_cloud[ip])) {
							this->qs[ip] = {
								{{Vec3{}, Vec{std::numeric_limits<float>::infinity()}}, {}},
								ObjSurfId::Null
							};
							continue;
						}

						std::pair<std::pair<Vec4, UVec3>, ObjSurfId> q{
							{{Vec3{}, Vec{std::numeric_limits<float>::infinity()}}, {}},
							ObjSurfId::Null
						};
						(
							[&]<surfacelike S_>(const std::vector<S_>& surf, const u8 isurf_kind) {
								for (usize isurf = 0; isurf < surf.size(); ++isurf) {
									const auto q_ = surf[isurf].closest_pdn(point_cloud[ip]);
									if (q_.first.w() < q.first.first.w()) {
										q = {q_, this->osids[isurf_kind][isurf]};
									}
								}
							}(std::get<idxs_>(this->moved_surfs), idxs_),
							...
						);

						this->qs[ip] = q;
					}
				}(std::index_sequence_for<Surfaces_...>{});

				// Ax = bのA, bを計算
				for (u8 iobj = 0; iobj < this->obj_num; ++iobj) {
					this->b[iobj] = Vec6{};
					this->a_w[iobj] = SymMat<3>{};
					this->a_t[iobj] = SymMat<3>{};
					this->a_wt[iobj] = SquareMat<3>{};
					this->counts[iobj] = 0;
				}
				for (usize ip = 0; ip < point_cloud.size(); ++ip) {
					const auto [qdn, osid] = this->qs[ip];
					if (osid == ObjSurfId::Null) continue;
					const auto [qd, n] = qdn;
					const auto q = qd.xyz();
					const auto d = qd.w();
					if (accept_distance2 < d) {
						this->qs[ip].second = ObjSurfId::Null;
						continue;
					}
					const Vec3 p = point_cloud[ip];
					const float err_n = vec::dot((p - q), n);
					const Vec3 p_c = vec::cross(p, n);

					const u8 iobj = std::to_underlying(osid_depack(osid).first);
					this->b[iobj] += Vec6{err_n * p_c, err_n * n};
					this->a_w[iobj] += vec::self_dyad(p_c);
					this->a_t[iobj] += vec::self_dyad(n);
					this->a_wt[iobj] += vec::dyad(p_c, n);
					this->counts[iobj]++;
				}
				float max_delta2 = 0.f;
				for (u8 iobj = 0; iobj < this->obj_num; ++iobj) {
					if (this->counts[iobj] < min_correspondences) {
						// 点が少なすぎるオブジェクトはスキップ
						this->obj_statuses[iobj] = ObjStatus::too_few_correspondences;
						continue;
					}
					this->b[iobj] /= this->counts[iobj];
					this->a_w[iobj] /= this->counts[iobj];
					this->a_t[iobj] /= this->counts[iobj];
					this->a_wt[iobj] /= this->counts[iobj];

					// tikhonovを足す(nによらない)
					this->a_w[iobj] += vec::diagonal_sym(tikhonov_w);
					this->a_t[iobj] += vec::diagonal_sym(tikhonov_t);

					// コレスキー分解、w, tを求める
					using Matrix6f = Eigen::Matrix<float, 6, 6>;

					Matrix6f a_tri;
					for (u8 i = 0; i < 3; ++i)
						for (u8 j = i; j < 3; ++j) {
							a_tri(i, j) = this->a_w[iobj][i, j];
							a_tri(i + 3, j + 3) = this->a_t[iobj][i, j];
						}
					for (u8 i = 0; i < 3; ++i)
						for (u8 j = 0; j < 3; ++j) { a_tri(i, j + 3) = this->a_wt[iobj][i, j]; }
					const Matrix6f a = a_tri.selfadjointView<Eigen::Upper>();
					Eigen::LLT<Matrix6f> cholesky(a);
					if (cholesky.info() != Eigen::Success) {
						this->obj_statuses[iobj] = ObjStatus::solve_failed;
						continue;
					}

					Eigen::Vector<float, 6> b_;
					for (u8 i = 0; i < 6; ++i) b_(i) = this->b[iobj][i];

					const auto x = cholesky.solve(b_);
					const SE3 diff =
						SE3{math::UQuaternion{vec::fast_normalize(Vec4{x[0], x[1], x[2], 2.f})},
							{x[3], x[4], x[5]}};

					// 推定姿勢を更新
					this->obj_poses[iobj] = (diff * this->obj_poses[iobj]).normalize();
					this->obj_statuses[iobj] = ObjStatus::updated;

					// 早期打ち切り判定用のdelta2 (w, tそれぞれのdotの和)
					const Vec3 w{x[0], x[1], x[2]};
					const Vec3 t{x[3], x[4], x[5]};
					const float delta2 = vec::dot(w, w) + vec::dot(t, t);
					if (max_delta2 < delta2) max_delta2 = delta2;
				}

				// 早期打ち切り: 姿勢を更新した全オブジェクトのdelta2の最大値が
				// convergence_delta2以下ならループを抜ける。max_loop_numがハード上限。
				if (max_delta2 <= convergence_delta2) break;
			}

			return IcpError::none;
		}
	};

	static_assert(icp_resource::icp_resource<
				  NormalKnownResource<ExplanationOnlySurface<0>, ExplanationOnlySurface<1>>,
				  ExplanationOnlySurface<0>,
				  ExplanationOnlySurface<1>>);
} // namespace sotoba::icp_resource::normal_known_icp_impl

namespace sotoba::icp_resource {
	using normal_known_icp_impl::NormalKnownResource;
}

#ifdef sotoba_ENABLE_TESTING
	#include <array>
	#include <cmath>
	#include <limits>
	#include <vector>

	#include <doctest.h>

	#include "sotoba/math/approx_check.hpp"

TEST_SUITE("normal_known_icp.hpp") {
	using namespace sotoba;
	using math::SE3;
	using math::UVec3;
	using math::Vec;
	using math::Vec3;
	using math::Vec4;
	using math::Vec6;
	namespace vec = math::vec;
	using surface::Rectangle;
	using icp_resource::IcpError;
	using icp_resource::NormalKnownResource;
	using icp_resource::ObjStatus;
	using math::ApproxCheck;

	// 原点(0,0,0)から見えるよう、ローカル座標系の中心を(0,0,0)に置いた矩形。
	// obj_poseで(0,0,5)へ移動させるとrectangle.hppのテストと同じ配置になる。
	inline auto forward_rect() -> Rectangle {
		return Rectangle{
			Vec3{0.f, 0.f, 0.f},
			Vec4{1.f, 0.f, 0.f, 2.f},
			Vec4{0.f, 1.f, 0.f, 1.f},
			UVec3{0.f, 0.f, -1.f},
		};
	}

	// 法線が原点と逆向きなので、obj_poseをどう動かしても可視化されない矩形。
	inline auto backward_rect() -> Rectangle {
		return Rectangle{
			Vec3{0.f, 0.f, 0.f},
			Vec4{1.f, 0.f, 0.f, 2.f},
			Vec4{0.f, 1.f, 0.f, 1.f},
			UVec3{0.f, 0.f, 1.f},
		};
	}

	// obj_num=1, 面1枚のNormalKnownResourceを作る
	inline auto make_icp(const Rectangle& rect, const usize capacity)
		-> NormalKnownResource<Rectangle> {
		std::array<std::vector<ObjSurfId>, 1> osids{
			std::vector<ObjSurfId>{osid_pack(ObjId(0), SurfId(0))}
		};
		return NormalKnownResource<Rectangle>{
			std::tuple{std::vector<Rectangle>{rect}},
			std::move(osids),
			1,
			capacity
		};
	}

	// 矩形ローカル座標系の面上の点を、true_poseでセンサ座標系へ写した
	// (ノイズ無しの)点群を作る。u:[-1.4,1.4], v:[-0.7,0.7] の範囲は
	// forward_rect() の半辺長(2.0, 1.0)に収まるのでクランプされない。
	inline auto sample_points(const SE3& true_pose) -> std::vector<Vec3> {
		std::vector<Vec3> pts;
		for (int iu = -2; iu <= 2; ++iu) {
			for (int iv = -2; iv <= 2; ++iv) {
				const Vec3 local{float(iu) * 0.7f, float(iv) * 0.35f, 0.f};
				pts.push_back(true_pose.app_v(local));
			}
		}
		return pts;
	}

	TEST_CASE("run_icp: 点数がちょうど容量ならIcpError::noneが返る") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f}));
		REQUIRE(points.size() == 25);

		const auto err = icp.run_icp(std::span{points}, Vec6{}, 1, 100.f);

		CHECK(err == IcpError::none);
	}

	TEST_CASE("run_icp: 点数が容量を1つでも超えるとtoo_many_pointsが返り状態が不変") {
		auto icp = make_icp(forward_rect(), 24);
		const auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f}));
		REQUIRE(points.size() == 25);

		const SE3 seed = SE3::trans(Vec3{0.1f, 0.f, 4.5f});
		icp.obj_pose(0) = seed;

		const auto err = icp.run_icp(std::span{points}, Vec6{}, 5, 100.f);

		CHECK(err == IcpError::too_many_points);
		CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		CHECK(icp.last_loop_count() == 0);
		CHECK(icp.obj_status(0) == ObjStatus::not_run);
	}

	TEST_CASE("run_icp: max_loop_num=0なら姿勢が変化せず回った回数は0") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f}));

		const SE3 seed = SE3::trans(Vec3{0.1f, 0.f, 4.5f});
		icp.obj_pose(0) = seed;

		const auto err = icp.run_icp(std::span{points}, Vec6{}, 0, 100.f);

		CHECK(err == IcpError::none);
		CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		CHECK(icp.last_loop_count() == 0);
	}

	TEST_CASE("run_icp: 回った回数は常にmax_loop_num以下") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f}));
		icp.obj_pose(0) = SE3::trans(Vec3{0.f, 0.f, 4.5f});

		const Vec6 tikhonov{0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f};
		constexpr u32 max_loop_num = 50;
		const auto err = icp.run_icp(std::span{points}, tikhonov, max_loop_num, 100.f);

		CHECK(err == IcpError::none);
		CHECK(icp.last_loop_count() <= max_loop_num);
	}

	TEST_CASE("run_icp: convergence_delta2を大きく与えるとmax_loop_numより少ない回数で打ち切られる") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f}));
		icp.obj_pose(0) = SE3::trans(Vec3{0.f, 0.f, 4.5f});

		const Vec6 tikhonov{0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f};
		constexpr u32 max_loop_num = 50;
		const auto err =
			icp.run_icp(std::span{points}, tikhonov, max_loop_num, 100.f, 1e6f);

		CHECK(err == IcpError::none);
		CHECK(icp.last_loop_count() < max_loop_num);
	}

	TEST_CASE("run_icp: 既知形状に対しずらしたシードが正解姿勢へ近づく") {
		auto icp = make_icp(forward_rect(), 25);
		const SE3 true_pose = SE3::trans(Vec3{0.f, 0.f, 5.f});
		const auto points = sample_points(true_pose);

		const SE3 seed = SE3::trans(Vec3{0.f, 0.f, 4.5f});
		icp.obj_pose(0) = seed;

		const Vec6 tikhonov{0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f};
		const auto err = icp.run_icp(std::span{points}, tikhonov, 50, 100.f);

		CHECK(err == IcpError::none);
		CHECK(icp.obj_status(0) == ObjStatus::updated);

		const float seed_err = std::fabs(seed.p.z() - true_pose.p.z());
		const float result_err = std::fabs(icp.obj_pose(0).p.z() - true_pose.p.z());
		CHECK(result_err < seed_err);
		CHECK(result_err < 0.05f);
	}

	TEST_CASE("run_icp: どの面にも対応しない点群はtoo_few_correspondencesになり姿勢はシードのまま") {
		auto icp = make_icp(backward_rect(), 4);
		const std::vector<Vec3> points{
			Vec3{0.f, 0.f, 1.f},
			Vec3{0.f, 0.f, 2.f},
			Vec3{0.f, 0.f, 3.f},
			Vec3{0.f, 0.f, 4.f},
		};

		const SE3 seed = SE3::trans(Vec3{1.f, 2.f, 3.f});
		icp.obj_pose(0) = seed;

		const auto err = icp.run_icp(std::span{points}, Vec6{}, 5, 100.f);

		CHECK(err == IcpError::none);
		CHECK(icp.obj_status(0) == ObjStatus::too_few_correspondences);
		CHECK(icp.correspondence_count(0) == 0);
		CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
	}

	TEST_CASE("run_icp: 非有限な点が混ざってもクラッシュせず対応点として採用されない") {
		auto icp = make_icp(forward_rect(), 27);
		auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f})); // 25点
		points.push_back(Vec3{std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f});
		points.push_back(Vec3{std::numeric_limits<float>::infinity(), 0.f, 0.f});
		REQUIRE(points.size() == 27);

		icp.obj_pose(0) = SE3::trans(Vec3{0.f, 0.f, 4.5f});

		const Vec6 tikhonov{0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f};
		const auto err = icp.run_icp(std::span{points}, tikhonov, 3, 100.f);

		CHECK(err == IcpError::none);
		CHECK(icp.qs[25].second == ObjSurfId::Null);
		CHECK(icp.qs[26].second == ObjSurfId::Null);
		CHECK(icp.correspondence_count(0) <= 25);
	}
}

#endif