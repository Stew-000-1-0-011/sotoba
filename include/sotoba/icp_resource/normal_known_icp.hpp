#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
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

#include "sotoba/math/se3.hpp"
#include "sotoba/math/square_mat.hpp"
#include "sotoba/math/vec.hpp"
#include "sotoba/surface/surface.hpp"

#include "resource.hpp"

namespace sotoba::icp_resource::normal_known_icp_impl {
	using math::SquareMat;
	using math::SymMat;
	using math::UVec3;
	using math::Vec;
	using math::Vec3;
	using math::Vec4;
	namespace vec = math::vec;
	using surface::ExplanationOnlySurface;
	using surface::surfacelike;
	using icp_resource::IcpError;
	using icp_resource::IcpParams;
	using icp_resource::IcpWeighting;
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
		std::vector<Sophus::SE3f::Tangent> b;
		std::vector<SymMat<3>> a_t;
		std::vector<SymMat<3>> a_w;
		std::vector<SquareMat<3>> a_tw;
		std::vector<usize> counts;
		std::vector<float> weight_sums;

		// ここに入れた姿勢をもとに、ICPがはしり、補正された結果がここに入る
		std::vector<Sophus::SE3f> obj_poses;

		std::vector<ObjStatus> obj_statuses;
		u32 loop_count;
		float last_gate2;

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
			, a_t{}
			, a_w{}
			, a_tw{}
			, counts{}
			, weight_sums{}
			, obj_poses{}
			, obj_statuses{}
			, loop_count{0}
			, last_gate2{0.f}
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
			this->a_t.resize(obj_num);
			this->a_w.resize(obj_num);
			this->a_tw.resize(obj_num);
			this->counts.resize(obj_num);
			this->weight_sums.resize(obj_num, 0.f);
			this->obj_poses.resize(obj_num, Sophus::SE3f{});
			this->obj_statuses.resize(obj_num, ObjStatus::not_run);
		}

		decltype(auto) obj_pose(this auto&& self, const u8 oid) noexcept {
			return self.obj_poses[oid];
		}

		auto points_capacity() const noexcept -> usize {
			return this->qs.size();
		}

		auto obj_status(const u8 oid) const noexcept -> ObjStatus {
			return this->obj_statuses[oid];
		}

		auto correspondence_count(const u8 oid) const noexcept -> usize {
			return this->counts[oid];
		}

		auto weight_sum(const u8 oid) const noexcept -> float {
			return this->weight_sums[oid];
		}

		auto last_loop_count() const noexcept -> u32 {
			return this->loop_count;
		}

		/// 最後の反復で使われたゲート。スケジュール有効時も accept_distance2 に一致する。
		auto last_accept_distance2() const noexcept -> float {
			return this->last_gate2;
		}

		/// 正規方程式の係数行列 A = Σ JᵀNJ。成分順序は Sophus::SE3f::Tangent と同じで、
		/// 添字 0..2 が並進、3..5 が回転。正規化も tikhonov も加えていない生の総和。
		auto information_matrix(const u8 oid) const noexcept -> Eigen::Matrix<float, 6, 6> {
			Eigen::Matrix<float, 6, 6> ret;
			for (u8 i = 0; i < 3; ++i)
				for (u8 j = 0; j < 3; ++j) {
					ret(i, j) = this->a_t[oid][i, j];
					ret(i + 3, j + 3) = this->a_w[oid][i, j];
					ret(i, j + 3) = this->a_tw[oid][i, j];
					ret(j + 3, i) = this->a_tw[oid][i, j];
				}
			return ret;
		}

		/// 正規方程式の右辺 b = Σ JᵀNe。こちらも生の総和で、成分順序は information_matrix と同じ。
		auto residual_vector(const u8 oid) const noexcept -> Sophus::SE3f::Tangent {
			return this->b[oid];
		}

		/// 対応点1つ = スカラー拘束1本なので、SE3 の6自由度には最低6点が要る。
		/// 必要条件にすぎず、これを満たしてもランク落ちはしうる。
		static constexpr usize min_correspondences = 6;

		/// 点対面ICPを走らせ、obj_poses を更新する。
		///
		/// 事前条件: point_cloud はセンサ座標系。可視性判定はセンサ原点(0,0,0)基準
		/// なので、obj_pose にはマップ座標系の形状をセンサ座標系へ写す変換を入れる。
		/// params.accept_distance2 系は距離の二乗。
		///
		/// 反復回数は params.max_loop_num 以下 (last_loop_count() で取得)。
		/// params.accept_distance2_begin > 0.f なら、1回目を accept_distance2_begin、
		/// 最終反復を accept_distance2 として等比でゲートを絞る。反復数は増えない。
		/// このとき早期打ち切りは最終反復でのみ判定される。
		///
		/// 以下は何も行わずに返し、姿勢・状態とも変化しない:
		/// - point_cloud.size() > points_capacity() → too_many_points (再確保しない)
		/// - max_loop_num == 0 → invalid_loop_num
		/// - accept_distance2 が正でない → invalid_accept_distance
		/// - priors の長さが 0 でも obj_num でもない → prior_size_mismatch
		/// - weighting が不正 → invalid_weighting
		/// - accept_distance2_begin が非有限、または accept_distance2 より小さい
		///   → invalid_accept_schedule
		auto run_icp(std::span<const Vec3> point_cloud, const IcpParams& params) noexcept
			-> IcpError {
			const auto& [max_loop_num,
						 accept_distance2,
						 convergence_delta2,
						 accept_distance2_begin,
						 tikhonov,
						 weighting,
						 priors] = params;

			if (point_cloud.size() > this->qs.size()) return IcpError::too_many_points;
			if (max_loop_num == 0) return IcpError::invalid_loop_num;
			if (!(accept_distance2 > 0.f)) return IcpError::invalid_accept_distance;
			if (!priors.empty() && priors.size() != this->obj_num)
				return IcpError::prior_size_mismatch;

			if (weighting.noise) {
				const auto& noise = *weighting.noise;
				if (!(noise.sigma_range >= 0.f) || !math::isfinite(noise.sigma_range))
					return IcpError::invalid_weighting;
				if (!(noise.sigma_angle >= 0.f) || !math::isfinite(noise.sigma_angle))
					return IcpError::invalid_weighting;
			}
			if (weighting.huber_k) {
				const float k = *weighting.huber_k;
				if (!(k > 0.f) || !math::isfinite(k)) return IcpError::invalid_weighting;
			}
			if (!math::isfinite(accept_distance2_begin))
				return IcpError::invalid_accept_schedule;
			if (accept_distance2_begin > 0.f && accept_distance2_begin < accept_distance2)
				return IcpError::invalid_accept_schedule;

			this->loop_count = 0;

			const bool scheduled = (accept_distance2_begin > 0.f) && (max_loop_num > 1);
			const float gate_ratio = scheduled
				? std::exp(
					  std::log(accept_distance2 / accept_distance2_begin)
					  / static_cast<float>(max_loop_num - 1)
				  )
				: 1.f;
			float gate2 = scheduled ? accept_distance2_begin : accept_distance2;

			for (u32 iloop = 0; iloop < max_loop_num; ++iloop) {
				this->loop_count = iloop + 1;

				// 不変条件: 最終反復のゲートは accept_distance2 に厳密に一致する
				const float current_gate2 =
					(iloop + 1 == max_loop_num) ? accept_distance2 : gate2;
				this->last_gate2 = current_gate2;

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
					this->b[iobj] = Sophus::SE3f::Tangent::Zero();
					this->a_t[iobj] = SymMat<3>{};
					this->a_w[iobj] = SymMat<3>{};
					this->a_tw[iobj] = SquareMat<3>{};
					this->counts[iobj] = 0;
					this->weight_sums[iobj] = 0.f;
				}
				for (usize ip = 0; ip < point_cloud.size(); ++ip) {
					const auto [qdn, osid] = this->qs[ip];
					if (osid == ObjSurfId::Null) continue;
					const auto [qd, n] = qdn;
					const auto q = qd.xyz();
					const auto d = qd.w();
					if (current_gate2 < d) {
						this->qs[ip].second = ObjSurfId::Null;
						continue;
					}
					const Vec3 p = point_cloud[ip];
					const float err_n = vec::dot((p - q), n);
					const Vec3 p_c = vec::cross(p, n);

					float w = 1.f;
					float sigma2 = 1.f; // 正規化残差を作るための分散 (noise 無指定なら 1)
					if (weighting.noise) {
						const float r2 = vec::dot(p, p);
						const float np = vec::dot(n, p); // |n| = 1 なので cos² = np²/r²
						const float cos2 = (r2 > float(math::epsilon)) ? (np * np / r2) : 1.f;
						const float sr2 = math::pow2(weighting.noise->sigma_range);
						const float st2 = math::pow2(weighting.noise->sigma_angle);
						sigma2 =
							std::max(sr2 * cos2 + r2 * st2 * (1.f - cos2), float(math::epsilon));
						w = 1.f / sigma2;
					}
					if (weighting.huber_k) {
						const float k = *weighting.huber_k;
						const float s2 = math::pow2(err_n) / sigma2; // 正規化残差の二乗
						if (math::pow2(k) < s2) { w *= k / math::sqrt(s2); }
					}

					const u8 iobj = std::to_underlying(osid_depack(osid).first);
					this->b[iobj].template head<3>() += (w * err_n) * math::to_eigen(n);
					this->b[iobj].template tail<3>() += (w * err_n) * math::to_eigen(p_c);
					this->a_t[iobj] += w * vec::self_dyad(n);
					this->a_w[iobj] += w * vec::self_dyad(p_c);
					this->a_tw[iobj] += w * vec::dyad(n, p_c);
					this->counts[iobj]++;
					this->weight_sums[iobj] += w;
				}

				gate2 *= gate_ratio;

				float max_delta2 = 0.f;
				for (u8 iobj = 0; iobj < this->obj_num; ++iobj) {
					if (this->counts[iobj] < min_correspondences) {
						this->obj_statuses[iobj] = ObjStatus::too_few_correspondences;
						continue;
					}
					// 不変条件: a_* / b は生の総和のまま。正規化と tikhonov はここでだけ適用する
					// 不変条件: 成分順序は Sophus::SE3f::Tangent = (並進, 回転)。a_t が左上、a_w が右下
					const float n = this->weight_sums[iobj];

					using Matrix6f = Eigen::Matrix<float, 6, 6>;

					Matrix6f a_tri;
					for (u8 i = 0; i < 3; ++i)
						for (u8 j = i; j < 3; ++j) {
							a_tri(i, j) =
								this->a_t[iobj][i, j] / n + (i == j ? tikhonov[i] : 0.f);
							a_tri(i + 3, j + 3) =
								this->a_w[iobj][i, j] / n + (i == j ? tikhonov[i + 3] : 0.f);
						}
					for (u8 i = 0; i < 3; ++i)
						for (u8 j = 0; j < 3; ++j) {
							a_tri(i, j + 3) = this->a_tw[iobj][i, j] / n;
						}
					const Matrix6f a = a_tri.selfadjointView<Eigen::Upper>();
					Eigen::LLT<Matrix6f> cholesky(a);
					if (cholesky.info() != Eigen::Success) {
						this->obj_statuses[iobj] = ObjStatus::solve_failed;
						continue;
					}

					const Sophus::SE3f::Tangent x = cholesky.solve(this->b[iobj] / n);

					this->obj_poses[iobj] = Sophus::SE3f::exp(x) * this->obj_poses[iobj];
					this->obj_poses[iobj].so3().normalize();
					this->obj_statuses[iobj] = ObjStatus::updated;

					const float delta2 = x.squaredNorm();
					if (max_delta2 < delta2) max_delta2 = delta2;
				}

				// 不変条件: 打ち切るのはゲートが accept_distance2 に達した反復のみ
				if (!scheduled || iloop + 1 == max_loop_num) {
					if (max_delta2 <= convergence_delta2) break;
				}
			}

			return IcpError::none;
		}
	};

	static_assert(icp_resource::icp_resource<
				  NormalKnownResource<ExplanationOnlySurface<0>, ExplanationOnlySurface<1>>,
				  ExplanationOnlySurface<0>,
				  ExplanationOnlySurface<1>>);
	static_assert(icp_resource::icp_resource<
				  NormalKnownResource<ExplanationOnlySurface<0>>,
				  ExplanationOnlySurface<0>>);
	static_assert(icp_resource::icp_resource<
				  NormalKnownResource<
					  ExplanationOnlySurface<0>,
					  ExplanationOnlySurface<1>,
					  ExplanationOnlySurface<2>>,
				  ExplanationOnlySurface<0>,
				  ExplanationOnlySurface<1>,
				  ExplanationOnlySurface<2>>);
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
	#include "sotoba/surface/box.hpp"
	// include 順に依存しないための自己完結用
	#include "sotoba/surface/rectangle.hpp"

TEST_SUITE("normal_known_icp.hpp") {
	using namespace sotoba;
	using math::UVec3;
	using math::Vec;
	using math::Vec3;
	using math::Vec4;
	using Tangent = Sophus::SE3f::Tangent;
	namespace vec = math::vec;
	using surface::BoxInner;
	using surface::Rectangle;
	using icp_resource::IcpError;
	using icp_resource::IcpParams;
	using icp_resource::IcpWeighting;
	using icp_resource::NoiseModel;
	using icp_resource::NormalKnownResource;
	using icp_resource::ObjStatus;
	using math::ApproxCheck;

	inline auto forward_rect() -> Rectangle {
		return Rectangle{
			Vec3{0.f, 0.f, 0.f},
			Vec4{1.f, 0.f, 0.f, 2.f},
			Vec4{0.f, 1.f, 0.f, 1.f},
			UVec3{0.f, 0.f, -1.f},
		};
	}

	inline auto backward_rect() -> Rectangle {
		return Rectangle{
			Vec3{0.f, 0.f, 0.f},
			Vec4{1.f, 0.f, 0.f, 2.f},
			Vec4{0.f, 1.f, 0.f, 1.f},
			UVec3{0.f, 0.f, 1.f},
		};
	}

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

	inline auto sample_points(const Sophus::SE3f& true_pose) -> std::vector<Vec3> {
		std::vector<Vec3> pts;
		for (int iu = -2; iu <= 2; ++iu) {
			for (int iv = -2; iv <= 2; ++iv) {
				const Vec3 local{float(iu) * 0.7f, float(iv) * 0.35f, 0.f};
				pts.push_back(math::app_v(true_pose, local));
			}
		}
		return pts;
	}

	TEST_CASE("run_icp: 点数がちょうど容量ならIcpError::noneが返る") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f}));
		REQUIRE(points.size() == 25);

		const auto err = icp.run_icp(
			std::span{points},
			{.max_loop_num = 1, .accept_distance2 = 100.f}
		);

		CHECK(err == IcpError::none);
	}

	TEST_CASE("run_icp: 点数が容量を1つでも超えるとtoo_many_pointsが返り状態が不変") {
		auto icp = make_icp(forward_rect(), 24);
		const auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f}));
		REQUIRE(points.size() == 25);

		const Sophus::SE3f seed = math::trans(Vec3{0.1f, 0.f, 4.5f});
		icp.obj_pose(0) = seed;

		const auto err = icp.run_icp(
			std::span{points},
			{.max_loop_num = 5, .accept_distance2 = 100.f}
		);

		CHECK(err == IcpError::too_many_points);
		CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		CHECK(icp.last_loop_count() == 0);
		CHECK(icp.obj_status(0) == ObjStatus::not_run);
	}

	TEST_CASE("run_icp: max_loop_num=0ならinvalid_loop_numが返り姿勢・状態が不変") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f}));

		const Sophus::SE3f seed = math::trans(Vec3{0.1f, 0.f, 4.5f});
		icp.obj_pose(0) = seed;

		const auto err = icp.run_icp(
			std::span{points},
			{.max_loop_num = 0, .accept_distance2 = 100.f}
		);

		CHECK(err == IcpError::invalid_loop_num);
		CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		CHECK(icp.obj_status(0) == ObjStatus::not_run);
		CHECK(icp.last_loop_count() == 0);
	}

	TEST_CASE("run_icp: accept_distance2が正でなければinvalid_accept_distanceが返り姿勢・状態が不変") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f}));

		const Sophus::SE3f seed = math::trans(Vec3{0.1f, 0.f, 4.5f});
		icp.obj_pose(0) = seed;

		auto check_rejected = [&](const IcpParams& params) {
			const auto err = icp.run_icp(std::span{points}, params);
			CHECK(err == IcpError::invalid_accept_distance);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
			CHECK(icp.obj_status(0) == ObjStatus::not_run);
			CHECK(icp.last_loop_count() == 0);
		};

		SUBCASE("0") { check_rejected({.max_loop_num = 5, .accept_distance2 = 0.f}); }

		SUBCASE("負") { check_rejected({.max_loop_num = 5, .accept_distance2 = -1.f}); }

		SUBCASE("NaN") {
			check_rejected(
				{.max_loop_num = 5,
				 .accept_distance2 = std::numeric_limits<float>::quiet_NaN()}
			);
		}

		SUBCASE("既定値のまま") { check_rejected({}); }
	}

	TEST_CASE("run_icp: priorsの長さが0でもobj_numでもなければprior_size_mismatchが返り姿勢・状態が不変") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f}));

		const Sophus::SE3f seed = math::trans(Vec3{0.1f, 0.f, 4.5f});
		icp.obj_pose(0) = seed;

		const std::vector<icp_resource::ObjPrior> priors(2);
		const auto err = icp.run_icp(
			std::span{points},
			{.max_loop_num = 5, .accept_distance2 = 100.f, .priors = std::span{priors}}
		);

		CHECK(err == IcpError::prior_size_mismatch);
		CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		CHECK(icp.obj_status(0) == ObjStatus::not_run);
		CHECK(icp.last_loop_count() == 0);
	}

	TEST_CASE("run_icp: 回った回数は常にmax_loop_num以下") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f}));
		icp.obj_pose(0) = math::trans(Vec3{0.f, 0.f, 4.5f});

		const Tangent tikhonov = Tangent::Constant(0.01f);
		constexpr u32 max_loop_num = 50;
		const auto err = icp.run_icp(
			std::span{points},
			{.max_loop_num = max_loop_num, .accept_distance2 = 100.f, .tikhonov = tikhonov}
		);

		CHECK(err == IcpError::none);
		CHECK(icp.last_loop_count() <= max_loop_num);
	}

	TEST_CASE("run_icp: convergence_delta2を大きく与えるとmax_loop_numより少ない回数で打ち切られる") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f}));
		icp.obj_pose(0) = math::trans(Vec3{0.f, 0.f, 4.5f});

		const Tangent tikhonov = Tangent::Constant(0.01f);
		constexpr u32 max_loop_num = 50;
		const auto err =
			icp.run_icp(
				std::span{points},
				{
					.max_loop_num = max_loop_num,
					.accept_distance2 = 100.f,
					.convergence_delta2 = 1e6f,
					.tikhonov = tikhonov
				}
			);

		CHECK(err == IcpError::none);
		CHECK(icp.last_loop_count() < max_loop_num);
	}

	TEST_CASE("run_icp: 既知形状に対しずらしたシードが正解姿勢へ近づく") {
		auto icp = make_icp(forward_rect(), 25);
		const Sophus::SE3f true_pose = math::trans(Vec3{0.f, 0.f, 5.f});
		const auto points = sample_points(true_pose);

		const Sophus::SE3f seed = math::trans(Vec3{0.f, 0.f, 4.5f});
		icp.obj_pose(0) = seed;

		const Tangent tikhonov = Tangent::Constant(0.001f);
		const auto err = icp.run_icp(
			std::span{points},
			{.max_loop_num = 50, .accept_distance2 = 100.f, .tikhonov = tikhonov}
		);

		CHECK(err == IcpError::none);
		CHECK(icp.obj_status(0) == ObjStatus::updated);

		const float seed_err = std::fabs(seed.translation().z() - true_pose.translation().z());
		const float result_err =
			std::fabs(icp.obj_pose(0).translation().z() - true_pose.translation().z());
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

		const Sophus::SE3f seed = math::trans(Vec3{1.f, 2.f, 3.f});
		icp.obj_pose(0) = seed;

		const auto err = icp.run_icp(
			std::span{points},
			{.max_loop_num = 5, .accept_distance2 = 100.f}
		);

		CHECK(err == IcpError::none);
		CHECK(icp.obj_status(0) == ObjStatus::too_few_correspondences);
		CHECK(icp.correspondence_count(0) == 0);
		CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
	}

	TEST_CASE("run_icp: 非有限な点が混ざってもクラッシュせず対応点として採用されない") {
		auto icp = make_icp(forward_rect(), 27);
		auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f})); // 25点
		points.push_back(Vec3{std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f});
		points.push_back(Vec3{std::numeric_limits<float>::infinity(), 0.f, 0.f});
		REQUIRE(points.size() == 27);

		icp.obj_pose(0) = math::trans(Vec3{0.f, 0.f, 4.5f});

		const Tangent tikhonov = Tangent::Constant(0.01f);
		const auto err = icp.run_icp(
			std::span{points},
			{.max_loop_num = 3, .accept_distance2 = 100.f, .tikhonov = tikhonov}
		);

		CHECK(err == IcpError::none);
		CHECK(icp.qs[25].second == ObjSurfId::Null);
		CHECK(icp.qs[26].second == ObjSurfId::Null);
		CHECK(icp.correspondence_count(0) <= 25);
	}

	TEST_CASE("information_matrix: tikhonovを変えても値が変わらない(正則化が焼き込まれていないこと)") {
		const auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f}));
		const Sophus::SE3f seed = math::trans(Vec3{0.05f, 0.f, 4.5f});

		auto icp_zero = make_icp(forward_rect(), 25);
		icp_zero.obj_pose(0) = seed;
		const auto err_zero = icp_zero.run_icp(
			std::span{points},
			{.max_loop_num = 1, .accept_distance2 = 100.f}
		);

		auto icp_big = make_icp(forward_rect(), 25);
		icp_big.obj_pose(0) = seed;
		const Tangent big_tikhonov = Tangent::Constant(1e6f);
		const auto err_big = icp_big.run_icp(
			std::span{points},
			{.max_loop_num = 1, .accept_distance2 = 100.f, .tikhonov = big_tikhonov}
		);

		REQUIRE(err_zero == IcpError::none);
		REQUIRE(err_big == IcpError::none);
		REQUIRE(icp_zero.correspondence_count(0) >= 3);
		REQUIRE(icp_big.correspondence_count(0) >= 3);

		const auto im_zero = icp_zero.information_matrix(0);
		const auto im_big = icp_big.information_matrix(0);
		for (u8 i = 0; i < 6; ++i)
			for (u8 j = i; j < 6; ++j) { CHECK(im_zero(i, j) == im_big(i, j)); }
	}

	TEST_CASE("run_icp: 対応点が min_correspondences 未満なら姿勢を更新しない") {
		static_assert(NormalKnownResource<Rectangle>::min_correspondences == 6);

		const Sophus::SE3f true_pose = math::trans(Vec3{0.f, 0.f, 5.f});
		const auto points = sample_points(true_pose); // 25点
		const Tangent tikhonov = Tangent::Constant(0.01f);

		SUBCASE("5点では更新されず、姿勢は呼び出し時の値のまま") {
			auto icp = make_icp(forward_rect(), 25);
			const Sophus::SE3f seed = math::trans(Vec3{0.f, 0.f, 4.9f});
			icp.obj_pose(0) = seed;

			const auto err =
				icp.run_icp(
					std::span{points}.subspan(0, 5),
					{.max_loop_num = 3, .accept_distance2 = 100.f, .tikhonov = tikhonov}
				);

			CHECK(err == IcpError::none);
			CHECK(icp.correspondence_count(0) == 5);
			CHECK(icp.obj_status(0) == ObjStatus::too_few_correspondences);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}

		SUBCASE("6点なら更新される") {
			auto icp = make_icp(forward_rect(), 25);
			const Sophus::SE3f seed = math::trans(Vec3{0.f, 0.f, 4.9f});
			icp.obj_pose(0) = seed;

			const auto err =
				icp.run_icp(
					std::span{points}.subspan(0, 6),
					{.max_loop_num = 3, .accept_distance2 = 100.f, .tikhonov = tikhonov}
				);

			CHECK(err == IcpError::none);
			CHECK(icp.correspondence_count(0) == 6);
			CHECK(icp.obj_status(0) == ObjStatus::updated);
			CHECK_FALSE(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}
	}

	TEST_CASE("information_matrix: 対称性") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f}));
		icp.obj_pose(0) = math::trans(Vec3{0.05f, 0.f, 4.5f});

		const Tangent tikhonov = Tangent::Constant(0.001f);
		const auto err = icp.run_icp(
			std::span{points},
			{.max_loop_num = 1, .accept_distance2 = 100.f, .tikhonov = tikhonov}
		);
		REQUIRE(err == IcpError::none);

		const auto im = icp.information_matrix(0);
		for (u8 i = 0; i < 6; ++i)
			for (u8 j = 0; j < 6; ++j) { CHECK(im(i, j) == im(j, i)); }
	}

	TEST_CASE("information_matrix: 正対した点群では生の総和がa_tブロックの手計算値と一致する") {
		auto icp = make_icp(forward_rect(), 25);
		const Sophus::SE3f true_pose = math::trans(Vec3{0.f, 0.f, 5.f});
		const auto points = sample_points(true_pose);
		icp.obj_pose(0) = true_pose; // シードなしで完全一致させる

		const auto err = icp.run_icp(
			std::span{points},
			{.max_loop_num = 1, .accept_distance2 = 100.f}
		);
		REQUIRE(err == IcpError::none);
		REQUIRE(icp.correspondence_count(0) == 25);

		const auto im = icp.information_matrix(0);
		CHECK(im(0, 0) == 0.f);
		CHECK(im(1, 1) == 0.f);
		CHECK(im(2, 2) == 25.f);
		CHECK(im(0, 1) == 0.f);
		CHECK(im(0, 2) == 0.f);
		CHECK(im(1, 2) == 0.f);

		const float n = float(icp.correspondence_count(0));
		CHECK(im(2, 2) / n > 0.f);
	}

	TEST_CASE("information_matrix: too_few_correspondencesでも生の総和(0除算やゴミではない)が読める") {
		auto icp = make_icp(backward_rect(), 4);
		const std::vector<Vec3> points{
			Vec3{0.f, 0.f, 1.f},
			Vec3{0.f, 0.f, 2.f},
			Vec3{0.f, 0.f, 3.f},
			Vec3{0.f, 0.f, 4.f},
		};
		icp.obj_pose(0) = math::trans(Vec3{1.f, 2.f, 3.f});

		const Tangent tikhonov = Tangent::Constant(10.f);
		const auto err = icp.run_icp(
			std::span{points},
			{.max_loop_num = 5, .accept_distance2 = 100.f, .tikhonov = tikhonov}
		);

		REQUIRE(err == IcpError::none);
		REQUIRE(icp.obj_status(0) == ObjStatus::too_few_correspondences);
		REQUIRE(icp.correspondence_count(0) == 0);

		const auto im = icp.information_matrix(0);
		for (u8 i = 0; i < 6; ++i)
			for (u8 j = i; j < 6; ++j) { CHECK(im(i, j) == 0.f); }

		const auto res = icp.residual_vector(0);
		for (u8 i = 0; i < 6; ++i) { CHECK(res[i] == 0.f); }
	}

	TEST_CASE("residual_vector: 読み取れる") {
		auto icp = make_icp(forward_rect(), 25);
		const Sophus::SE3f true_pose = math::trans(Vec3{0.f, 0.f, 5.f});
		const auto points = sample_points(true_pose);
		icp.obj_pose(0) = true_pose; // 完全一致 -> 各点の誤差は0

		const auto err = icp.run_icp(
			std::span{points},
			{.max_loop_num = 1, .accept_distance2 = 100.f}
		);
		REQUIRE(err == IcpError::none);
		REQUIRE(icp.correspondence_count(0) == 25);

		const auto res = icp.residual_vector(0);
		for (u8 i = 0; i < 6; ++i) { CHECK(res[i] == doctest::Approx(0.f).epsilon(1e-4)); }
	}


	TEST_CASE("run_icp: 既定のIcpWeighting{}ではweight_sumが対応点数と厳密に一致する") {
		auto icp = make_icp(forward_rect(), 25);
		const Sophus::SE3f true_pose = math::trans(Vec3{0.f, 0.f, 5.f});
		const auto points = sample_points(true_pose);
		icp.obj_pose(0) = math::trans(Vec3{0.05f, 0.f, 4.5f});

		const Tangent tikhonov = Tangent::Constant(0.001f);
		const auto err = icp.run_icp(
			std::span{points},
			{.max_loop_num = 1, .accept_distance2 = 100.f, .tikhonov = tikhonov}
		);

		REQUIRE(err == IcpError::none);
		REQUIRE(icp.correspondence_count(0) == 25);
		CHECK(icp.weight_sum(0) == float(icp.correspondence_count(0)));
	}

	TEST_CASE("run_icp: ノイズモデルが入射角で効く(正対 vs 斜め)") {
		auto trace_of = [](const Rectangle& rect,
						   const Sophus::SE3f& true_pose,
						   const float sigma_angle) {
			auto icp = make_icp(rect, 25);
			const auto points = sample_points(true_pose);
			icp.obj_pose(0) = true_pose; // シードなしで完全一致 (残差0でも重みは効く)

			const IcpWeighting weighting{
				.noise = NoiseModel{.sigma_range = 0.05f, .sigma_angle = sigma_angle}
			};
			const auto err = icp.run_icp(
				std::span{points},
				{.max_loop_num = 1, .accept_distance2 = 100.f, .weighting = weighting}
			);
			REQUIRE(err == IcpError::none);
			REQUIRE(icp.correspondence_count(0) >= 3);

			const auto im = icp.information_matrix(0);
			float trace = 0.f;
			for (u8 i = 0; i < 6; ++i) trace += im(i, i);
			return trace;
		};

		const Sophus::SE3f straight_pose = math::trans(Vec3{0.f, 0.f, 5.f});
		const Sophus::SE3f tilted_pose =
			Sophus::SE3f{math::ypr(Vec3{0.f, 0.9f, 0.f}), Eigen::Vector3f{0.f, 0.f, 5.f}};

		const float trace_straight_0 = trace_of(forward_rect(), straight_pose, 0.f);
		const float trace_straight_1 = trace_of(forward_rect(), straight_pose, 0.5f);
		const float trace_tilted_0 = trace_of(forward_rect(), tilted_pose, 0.f);
		const float trace_tilted_1 = trace_of(forward_rect(), tilted_pose, 0.5f);

		REQUIRE(trace_straight_0 > 0.f);
		REQUIRE(trace_tilted_0 > 0.f);

		const float ratio_straight = trace_straight_1 / trace_straight_0;
		const float ratio_tilted = trace_tilted_1 / trace_tilted_0;

		CHECK(ratio_tilted < ratio_straight);
	}

	TEST_CASE("run_icp: ノイズモデルで遠い点の重みが落ちる(grazing)") {
		auto icp0 = make_icp(forward_rect(), 25);
		auto icp1 = make_icp(forward_rect(), 25);

		const Sophus::SE3f grazing_pose =
			Sophus::SE3f{math::ypr(Vec3{0.f, 1.0f, 0.f}), Eigen::Vector3f{0.f, 0.f, 5.f}};
		const auto points = sample_points(grazing_pose);
		icp0.obj_pose(0) = grazing_pose;
		icp1.obj_pose(0) = grazing_pose;

		const IcpWeighting weighting0{.noise = NoiseModel{.sigma_range = 0.05f, .sigma_angle = 0.f}
		};
		const IcpWeighting weighting1{
			.noise = NoiseModel{.sigma_range = 0.05f, .sigma_angle = 0.3f}
		};

		const auto err0 = icp0.run_icp(
			std::span{points},
			{.max_loop_num = 1, .accept_distance2 = 100.f, .weighting = weighting0}
		);
		const auto err1 = icp1.run_icp(
			std::span{points},
			{.max_loop_num = 1, .accept_distance2 = 100.f, .weighting = weighting1}
		);
		REQUIRE(err0 == IcpError::none);
		REQUIRE(err1 == IcpError::none);
		REQUIRE(icp0.correspondence_count(0) >= 3);
		REQUIRE(icp1.correspondence_count(0) >= 3);

		const auto im0 = icp0.information_matrix(0);
		const auto im1 = icp1.information_matrix(0);
		float trace0 = 0.f, trace1 = 0.f;
		for (u8 i = 0; i < 6; ++i) {
			trace0 += im0(i, i);
			trace1 += im1(i, i);
		}
		CHECK(trace1 < trace0);
		CHECK(icp1.weight_sum(0) < icp0.weight_sum(0));
	}

	TEST_CASE("run_icp: 【本命】Huberが外れ値に効く") {
		const Sophus::SE3f true_pose = math::trans(Vec3{0.f, 0.f, 5.f});
		auto points = sample_points(true_pose); // 25点、正しい点群
		points.push_back(math::app_v(true_pose, Vec3{0.5f, 0.3f, 2.0f}));
		points.push_back(math::app_v(true_pose, Vec3{-0.5f, -0.3f, 2.0f}));
		points.push_back(math::app_v(true_pose, Vec3{0.0f, 0.6f, 2.0f}));
		REQUIRE(points.size() == 28);

		const Sophus::SE3f seed = math::trans(Vec3{0.f, 0.f, 4.5f});
		const Tangent tikhonov = Tangent::Constant(0.001f);
		constexpr u32 max_loop_num = 30;

		auto icp_no_huber = make_icp(forward_rect(), 28);
		icp_no_huber.obj_pose(0) = seed;
		const auto err_no_huber =
			icp_no_huber.run_icp(
				std::span{points},
				{.max_loop_num = max_loop_num, .accept_distance2 = 100.f, .tikhonov = tikhonov}
			);

		auto icp_huber = make_icp(forward_rect(), 28);
		icp_huber.obj_pose(0) = seed;
		const IcpWeighting weighting{.huber_k = 0.1f};
		const auto err_huber = icp_huber.run_icp(
			std::span{points},
			{
				.max_loop_num = max_loop_num,
				.accept_distance2 = 100.f,
				.tikhonov = tikhonov,
				.weighting = weighting
			}
		);

		REQUIRE(err_no_huber == IcpError::none);
		REQUIRE(err_huber == IcpError::none);

		const float err_z_no_huber =
			std::fabs(icp_no_huber.obj_pose(0).translation().z() - true_pose.translation().z());
		const float err_z_huber =
			std::fabs(icp_huber.obj_pose(0).translation().z() - true_pose.translation().z());

		CHECK(err_z_no_huber > 0.1f);
		CHECK(err_z_huber < 0.05f);
		CHECK(err_z_huber < err_z_no_huber);
	}

	TEST_CASE("run_icp: weightingが不正ならinvalid_weightingが返り姿勢が不変") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f}));
		const Sophus::SE3f seed = math::trans(Vec3{0.f, 0.f, 4.5f});

		SUBCASE("sigma_rangeが負") {
			icp.obj_pose(0) = seed;
			const IcpWeighting weighting{
				.noise = NoiseModel{.sigma_range = -0.1f, .sigma_angle = 0.f}
			};
			const auto err = icp.run_icp(
				std::span{points},
				{.max_loop_num = 5, .accept_distance2 = 100.f, .weighting = weighting}
			);
			CHECK(err == IcpError::invalid_weighting);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
			CHECK(icp.obj_status(0) == ObjStatus::not_run);
		}

		SUBCASE("sigma_angleが負") {
			icp.obj_pose(0) = seed;
			const IcpWeighting weighting{
				.noise = NoiseModel{.sigma_range = 0.f, .sigma_angle = -0.1f}
			};
			const auto err = icp.run_icp(
				std::span{points},
				{.max_loop_num = 5, .accept_distance2 = 100.f, .weighting = weighting}
			);
			CHECK(err == IcpError::invalid_weighting);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}

		SUBCASE("huber_kが0") {
			icp.obj_pose(0) = seed;
			const IcpWeighting weighting{.huber_k = 0.f};
			const auto err = icp.run_icp(
				std::span{points},
				{.max_loop_num = 5, .accept_distance2 = 100.f, .weighting = weighting}
			);
			CHECK(err == IcpError::invalid_weighting);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}

		SUBCASE("huber_kが負") {
			icp.obj_pose(0) = seed;
			const IcpWeighting weighting{.huber_k = -1.f};
			const auto err = icp.run_icp(
				std::span{points},
				{.max_loop_num = 5, .accept_distance2 = 100.f, .weighting = weighting}
			);
			CHECK(err == IcpError::invalid_weighting);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}

		SUBCASE("sigma_rangeがNaN") {
			icp.obj_pose(0) = seed;
			const IcpWeighting weighting{
				.noise =
					NoiseModel{
						.sigma_range = std::numeric_limits<float>::quiet_NaN(), .sigma_angle = 0.f
					}
			};
			const auto err = icp.run_icp(
				std::span{points},
				{.max_loop_num = 5, .accept_distance2 = 100.f, .weighting = weighting}
			);
			CHECK(err == IcpError::invalid_weighting);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}

		SUBCASE("huber_kがNaN") {
			icp.obj_pose(0) = seed;
			const IcpWeighting weighting{.huber_k = std::numeric_limits<float>::quiet_NaN()};
			const auto err = icp.run_icp(
				std::span{points},
				{.max_loop_num = 5, .accept_distance2 = 100.f, .weighting = weighting}
			);
			CHECK(err == IcpError::invalid_weighting);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}
	}

	TEST_CASE("run_icp: sigma_range=0,sigma_angle=0でもNaN/infにならない(epsilonガード)") {
		auto icp = make_icp(forward_rect(), 25);
		const Sophus::SE3f true_pose = math::trans(Vec3{0.f, 0.f, 5.f});
		const auto points = sample_points(true_pose);
		icp.obj_pose(0) = math::trans(Vec3{0.05f, 0.f, 4.5f});

		const Tangent tikhonov = Tangent::Constant(0.001f);
		const IcpWeighting weighting{.noise = NoiseModel{.sigma_range = 0.f, .sigma_angle = 0.f}};
		const auto err = icp.run_icp(
			std::span{points},
			{
				.max_loop_num = 3,
				.accept_distance2 = 100.f,
				.tikhonov = tikhonov,
				.weighting = weighting
			}
		);

		CHECK(err == IcpError::none);
		CHECK(icp.obj_pose(0).translation().allFinite());
		CHECK(std::isfinite(icp.weight_sum(0)));
		const auto im = icp.information_matrix(0);
		for (u8 i = 0; i < 6; ++i)
			for (u8 j = i; j < 6; ++j) { CHECK(std::isfinite(im(i, j))); }
	}


	inline auto box_hlens() -> Vec3 { return Vec3{2.f, 2.f, 2.f}; }

	inline auto make_box_icp(const usize capacity) -> NormalKnownResource<BoxInner> {
		const BoxInner local_box{Vec3{0.f, 0.f, 0.f}, math::SquareMat<3>::ide(), box_hlens()};
		std::array<std::vector<ObjSurfId>, 1> osids{
			std::vector<ObjSurfId>{osid_pack(ObjId(0), SurfId(0))}
		};
		return NormalKnownResource<BoxInner>{
			std::tuple{std::vector<BoxInner>{local_box}},
			std::move(osids),
			1,
			capacity
		};
	}

	inline auto sample_box_points(const Sophus::SE3f& true_pose) -> std::vector<Vec3> {
		const Vec3 hlens = box_hlens();
		std::vector<Vec3> local_pts;
		for (int axis = 0; axis < 3; ++axis) {
			const int j = (axis + 1) % 3;
			const int k = (axis + 2) % 3;
			for (const float sign : {-1.f, 1.f}) {
				for (const float fj : {-0.6f, 0.f, 0.6f}) {
					for (const float fk : {-0.6f, 0.f, 0.6f}) {
						Vec3 local{};
						local[axis] = sign * hlens[axis];
						local[j] = fj * hlens[j];
						local[k] = fk * hlens[k];
						local_pts.push_back(local);
					}
				}
			}
		}
		std::vector<Vec3> pts;
		pts.reserve(local_pts.size());
		for (const auto& lp : local_pts) pts.push_back(math::app_v(true_pose, lp));
		return pts;
	}

	TEST_CASE(
		"run_icp: 【本命】coarse-to-fineで同じmax_loop_numのまま収束半径が広がる(BoxInner)"
	) {
		const Sophus::SE3f true_pose = Sophus::SE3f{};
		const auto points = sample_box_points(true_pose);
		REQUIRE(points.size() == 54);

		const Sophus::SE3f seed = math::trans(Vec3{0.35f, 0.35f, 0.35f});

		const Tangent tikhonov = Tangent::Constant(0.001f);
		constexpr u32 max_loop_num = 30;
		constexpr float accept_distance2 = 0.04f; // 0.2m: シードの誤差(0.35m)より狭い
		constexpr float accept_distance2_begin = 4.0f; // 2.0m: シードの誤差より十分広い
		constexpr float convergence_delta2 = -1.f;

		auto icp_no_schedule = make_box_icp(points.size());
		icp_no_schedule.obj_pose(0) = seed;
		const auto err_no_schedule = icp_no_schedule.run_icp(
			std::span{points},
			{
				.max_loop_num = max_loop_num,
				.accept_distance2 = accept_distance2,
				.convergence_delta2 = convergence_delta2,
				.tikhonov = tikhonov
			}
		);

		auto icp_scheduled = make_box_icp(points.size());
		icp_scheduled.obj_pose(0) = seed;
		const auto err_scheduled = icp_scheduled.run_icp(
			std::span{points},
			{
				.max_loop_num = max_loop_num,
				.accept_distance2 = accept_distance2,
				.convergence_delta2 = convergence_delta2,
				.accept_distance2_begin = accept_distance2_begin,
				.tikhonov = tikhonov
			}
		);

		REQUIRE(err_no_schedule == IcpError::none);
		REQUIRE(err_scheduled == IcpError::none);

		CHECK(icp_no_schedule.last_loop_count() == max_loop_num);
		CHECK(icp_scheduled.last_loop_count() == max_loop_num);
		CHECK(icp_no_schedule.last_loop_count() == icp_scheduled.last_loop_count());

		CHECK(icp_no_schedule.obj_status(0) == ObjStatus::too_few_correspondences);
		CHECK(icp_no_schedule.correspondence_count(0) == 0);
		CHECK(ApproxCheck{icp_no_schedule.obj_pose(0)} == ApproxCheck{seed});

		CHECK(icp_scheduled.obj_status(0) == ObjStatus::updated);

		const float err_no_schedule2 =
			(icp_no_schedule.obj_pose(0).translation() - true_pose.translation()).squaredNorm();
		const float err_scheduled2 =
			(icp_scheduled.obj_pose(0).translation() - true_pose.translation()).squaredNorm();
		CHECK(err_scheduled2 < 0.01f); // 並進誤差 < 10cm まで収束する

		CHECK(err_scheduled2 < err_no_schedule2);
	}

	TEST_CASE("run_icp: 最終反復のゲートがaccept_distance2と厳密に一致する(境界の外れ点で間接確認)") {
		const Sophus::SE3f true_pose = math::trans(Vec3{0.f, 0.f, 5.f});
		auto points = sample_points(true_pose); // 25点、すべて面上ぴったり
		points.push_back(math::app_v(true_pose, Vec3{0.f, 0.f, 1.0f}));
		REQUIRE(points.size() == 26);

		constexpr float accept_distance2 = 0.04f; // 0.2m
		constexpr float accept_distance2_begin = 4.0f; // 2.0m
		constexpr u32 max_loop_num = 5;
		const Tangent tikhonov = Tangent::Constant(0.001f);

		auto icp_no_schedule = make_icp(forward_rect(), points.size());
		icp_no_schedule.obj_pose(0) = true_pose; // 既に正解姿勢
		const auto err_no_schedule =
			icp_no_schedule.run_icp(
				std::span{points},
				{
					.max_loop_num = max_loop_num,
					.accept_distance2 = accept_distance2,
					.tikhonov = tikhonov
				}
			);

		auto icp_scheduled = make_icp(forward_rect(), points.size());
		icp_scheduled.obj_pose(0) = true_pose;
		const auto err_scheduled = icp_scheduled.run_icp(
			std::span{points},
			{
				.max_loop_num = max_loop_num,
				.accept_distance2 = accept_distance2,
				.accept_distance2_begin = accept_distance2_begin,
				.tikhonov = tikhonov
			}
		);

		REQUIRE(err_no_schedule == IcpError::none);
		REQUIRE(err_scheduled == IcpError::none);

		CHECK(icp_no_schedule.correspondence_count(0) == 25);
		CHECK(icp_scheduled.correspondence_count(0) == icp_no_schedule.correspondence_count(0));
	}

	TEST_CASE("run_icp: スケジュール有効でも最終反復のゲートはaccept_distance2と厳密に一致する") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f}));
		icp.obj_pose(0) = math::trans(Vec3{0.f, 0.f, 4.9f});

		constexpr float accept2 = 0.03f;
		constexpr float begin2 = 7.f;
		constexpr u32 max_loop_num = 17;
		const Tangent tikhonov = Tangent::Constant(0.01f);

		const auto err =
			icp.run_icp(
				std::span{points},
				{
					.max_loop_num = max_loop_num,
					.accept_distance2 = accept2,
					.accept_distance2_begin = begin2,
					.tikhonov = tikhonov
				}
			);

		REQUIRE(err == IcpError::none);
		REQUIRE(icp.last_loop_count() == max_loop_num);
		CHECK(icp.last_accept_distance2() == accept2);

		float acc = begin2;
		const float ratio =
			std::exp(std::log(accept2 / begin2) / static_cast<float>(max_loop_num - 1));
		for (u32 i = 0; i + 1 < max_loop_num; ++i) acc *= ratio;
		CHECK(acc != accept2);
	}

	TEST_CASE("run_icp: max_loop_num=1でスケジュールを指定してもaccept_distance2が使われ壊れない") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f}));
		icp.obj_pose(0) = math::trans(Vec3{0.f, 0.f, 4.5f});

		const auto err =
			icp.run_icp(
				std::span{points},
				{.max_loop_num = 1, .accept_distance2 = 100.f, .accept_distance2_begin = 200.f}
			);

		CHECK(err == IcpError::none);
		CHECK(icp.last_loop_count() == 1);
	}

	TEST_CASE("run_icp: accept_distance2_beginが不正ならinvalid_accept_scheduleが返り姿勢が不変") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f}));
		const Sophus::SE3f seed = math::trans(Vec3{0.f, 0.f, 4.5f});

		SUBCASE("accept_distance2より小さい(狭い→広いの逆順)") {
			icp.obj_pose(0) = seed;
			const auto err =
				icp.run_icp(
					std::span{points},
					{.max_loop_num = 5, .accept_distance2 = 100.f, .accept_distance2_begin = 50.f}
				);
			CHECK(err == IcpError::invalid_accept_schedule);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
			CHECK(icp.obj_status(0) == ObjStatus::not_run);
			CHECK(icp.last_loop_count() == 0);
		}

		SUBCASE("NaN") {
			icp.obj_pose(0) = seed;
			const auto err = icp.run_icp(
				std::span{points},
				{
					.max_loop_num = 5,
					.accept_distance2 = 100.f,
					.accept_distance2_begin = std::numeric_limits<float>::quiet_NaN()
				}
			);
			CHECK(err == IcpError::invalid_accept_schedule);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}

		SUBCASE("+inf") {
			icp.obj_pose(0) = seed;
			const auto err = icp.run_icp(
				std::span{points},
				{
					.max_loop_num = 5,
					.accept_distance2 = 100.f,
					.accept_distance2_begin = std::numeric_limits<float>::infinity()
				}
			);
			CHECK(err == IcpError::invalid_accept_schedule);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}

		SUBCASE("-inf") {
			icp.obj_pose(0) = seed;
			const auto err = icp.run_icp(
				std::span{points},
				{
					.max_loop_num = 5,
					.accept_distance2 = 100.f,
					.accept_distance2_begin = -std::numeric_limits<float>::infinity()
				}
			);
			CHECK(err == IcpError::invalid_accept_schedule);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}
	}

	TEST_CASE("run_icp: スケジュール有効時はconvergence_delta2を非常に大きくしてもmax_loop_numまで回る") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(math::trans(Vec3{0.f, 0.f, 5.f}));
		icp.obj_pose(0) = math::trans(Vec3{0.f, 0.f, 4.5f});

		const Tangent tikhonov = Tangent::Constant(0.01f);
		constexpr u32 max_loop_num = 20;
		const auto err = icp.run_icp(
			std::span{points},
			{
				.max_loop_num = max_loop_num,
				.accept_distance2 = 100.f,
				.convergence_delta2 = 1e6f,
				.accept_distance2_begin = 400.f,
				.tikhonov = tikhonov
			}
		);

		CHECK(err == IcpError::none);
		CHECK(icp.last_loop_count() == max_loop_num);
	}

	TEST_CASE("information_matrix: 各点のヤコビアン [n; p×n] からの Σ JᵀJ と一致する(並進が先、回転が後)") {
		auto icp = make_icp(forward_rect(), 25);
		const Sophus::SE3f pose{
			math::ypr(Vec3{0.3f, 0.6f, -0.4f}),
			Eigen::Vector3f{0.2f, -0.1f, 5.f}
		};
		const auto points = sample_points(pose);
		icp.obj_pose(0) = pose;

		const auto err =
			icp.run_icp(std::span{points}, {.max_loop_num = 1, .accept_distance2 = 100.f});
		REQUIRE(err == IcpError::none);
		REQUIRE(icp.correspondence_count(0) == 25);

		const Eigen::Vector3f n = pose.so3() * Eigen::Vector3f{0.f, 0.f, -1.f};
		Eigen::Matrix<float, 6, 6> expected = Eigen::Matrix<float, 6, 6>::Zero();
		for (const auto& p : points) {
			Tangent j;
			j << n, math::to_eigen(p).cross(n);
			expected += j * j.transpose();
		}

		const auto im = icp.information_matrix(0);
		const float scale = expected.cwiseAbs().maxCoeff();
		for (u8 i = 0; i < 6; ++i)
			for (u8 j = 0; j < 6; ++j) {
				CHECK(std::fabs(im(i, j) - expected(i, j)) < 1e-4f * scale);
			}
		CHECK(expected.topRightCorner<3, 3>().cwiseAbs().maxCoeff() > 1e-2f * scale);
	}

	TEST_CASE("run_icp: 1反復の解が真の左摂動の逆を(並進, 回転)の順で返す") {
		const Sophus::SE3f true_pose{
			math::ypr(Vec3{0.2f, 0.15f, 0.1f}),
			Eigen::Vector3f{0.3f, -0.2f, 0.1f}
		};
		const auto points = sample_box_points(true_pose);

		Tangent xi0;
		xi0 << 0.03f, -0.02f, 0.01f, 0.02f, -0.03f, 0.025f;
		const Sophus::SE3f seed = Sophus::SE3f::exp(xi0) * true_pose;

		auto icp = make_box_icp(points.size());
		icp.obj_pose(0) = seed;
		const auto err =
			icp.run_icp(std::span{points}, {.max_loop_num = 1, .accept_distance2 = 100.f});
		REQUIRE(err == IcpError::none);
		REQUIRE(icp.obj_status(0) == ObjStatus::updated);

		const Tangent x = (icp.obj_pose(0) * seed.inverse()).log();
		for (u8 i = 0; i < 6; ++i) { CHECK(std::fabs(x[i] + xi0[i]) < 2e-3f); }
	}

	TEST_CASE("residual_vector: 左摂動 ξ0 に対し b ≈ -A ξ0 (成分順序が information_matrix と一致する)") {
		const Sophus::SE3f true_pose{
			math::ypr(Vec3{0.2f, 0.15f, 0.1f}),
			Eigen::Vector3f{0.3f, -0.2f, 0.1f}
		};
		const auto points = sample_box_points(true_pose);

		Tangent xi0;
		xi0 << 0.01f, -0.008f, 0.006f, 0.007f, -0.01f, 0.009f;

		auto icp = make_box_icp(points.size());
		icp.obj_pose(0) = Sophus::SE3f::exp(xi0) * true_pose;
		const auto err =
			icp.run_icp(std::span{points}, {.max_loop_num = 1, .accept_distance2 = 100.f});
		REQUIRE(err == IcpError::none);

		const Tangent expected = -(icp.information_matrix(0) * xi0);
		const Tangent b = icp.residual_vector(0);
		CHECK((b - expected).norm() < 0.05f * expected.norm());
	}

	TEST_CASE("run_icp: tikhonovは(並進, 回転)の順で効く") {
		const Sophus::SE3f true_pose{
			math::ypr(Vec3{0.2f, 0.15f, 0.1f}),
			Eigen::Vector3f{0.3f, -0.2f, 0.1f}
		};
		const auto points = sample_box_points(true_pose);

		Tangent xi0;
		xi0 << 0.05f, -0.04f, 0.03f, 0.04f, -0.05f, 0.03f;
		const Sophus::SE3f seed = Sophus::SE3f::exp(xi0) * true_pose;

		auto step = [&](const Tangent& tikhonov) {
			auto icp = make_box_icp(points.size());
			icp.obj_pose(0) = seed;
			const auto err = icp.run_icp(
				std::span{points},
				{.max_loop_num = 1, .accept_distance2 = 100.f, .tikhonov = tikhonov}
			);
			REQUIRE(err == IcpError::none);
			REQUIRE(icp.obj_status(0) == ObjStatus::updated);
			return Tangent{(icp.obj_pose(0) * seed.inverse()).log()};
		};

		Tangent lock_translation = Tangent::Zero();
		lock_translation.head<3>().setConstant(1e6f);
		const Tangent x_rot_only = step(lock_translation);
		CHECK(x_rot_only.head<3>().norm() < 1e-4f);
		CHECK(x_rot_only.tail<3>().norm() > 1e-2f);

		Tangent lock_rotation = Tangent::Zero();
		lock_rotation.tail<3>().setConstant(1e6f);
		const Tangent x_trans_only = step(lock_rotation);
		CHECK(x_trans_only.tail<3>().norm() < 1e-4f);
		CHECK(x_trans_only.head<3>().norm() > 1e-2f);
	}
}

#endif