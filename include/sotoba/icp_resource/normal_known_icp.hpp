#pragma once

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
#ifndef sotoba_USE_SYCL
	using Vec6 = Vec<6>;
#else
	using Vec6 = Vec<8>;
#endif
	namespace vec = math::vec;
	using surface::ExplanationOnlySurface;
	using surface::surfacelike;

	template <surfacelike... Surfaces_>
	struct NormalKnownNonSyclResource final {
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

		u8 obj_num;

		NormalKnownNonSyclResource(
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
		}

		decltype(auto) obj_pose(this auto&& self, const u8 oid) noexcept {
			return self.obj_poses[oid];
		}

		void run_icp(
			std::vector<Vec3>&& point_cloud,
			const Vec6& tikhonov,
			const u32 loop_num,
			const float accept_distance2
		) noexcept {
			const auto tikhonov_w = vec::split<0, 3>(tikhonov);
			const auto tikhonov_t = vec::split<3, 6>(tikhonov);

			for (u32 iloop = 0; iloop < loop_num; ++iloop) {
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
						}

						std::pair<std::pair<Vec4, UVec3>, ObjSurfId> q{
							{{Vec3{}, Vec{std::numeric_limits<float>::infinity()}}, {}},
							ObjSurfId::Null
						};
						(
							[&]<surfacelike S_>(const std::vector<S_>& surf, const u8 isurf_kind) {
								for (u8 isurf = 0; isurf < surf.size(); ++isurf) {
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
#ifndef sotoba_USE_SYCL
					this->b[iobj] += Vec6{err_n * p_c, err_n * n};
#else
					this->b[iobj] += Vec6{err_n * p_c, err_n * n, 0.f, 0.f};
#endif
					this->a_w[iobj] += vec::self_dyad(p_c);
					this->a_t[iobj] += vec::self_dyad(n);
					this->a_wt[iobj] += vec::dyad(p_c, n);
					this->counts[iobj]++;
				}
				for (u8 iobj = 0; iobj < this->obj_num; ++iobj) {
					if (this->counts[iobj] < 3) continue; // 点が少なすぎるオブジェクトはスキップ
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
					if (cholesky.info() != Eigen::Success) { continue; }

					Eigen::Vector<float, 6> b_;
					for (u8 i = 0; i < 6; ++i) b_(i) = this->b[iobj][i];

					const auto x = cholesky.solve(b_);
					const SE3 diff =
						SE3{math::UQuaternion{vec::fast_normalize(Vec4{x[0], x[1], x[2], 2.f})},
							{x[3], x[4], x[5]}};

					// 推定姿勢を更新
					this->obj_poses[iobj] = (diff * this->obj_poses[iobj]).normalize();
				}
			}
		}
	};

	static_assert(icp_resource::icp_resource<
				  NormalKnownNonSyclResource<ExplanationOnlySurface<0>, ExplanationOnlySurface<1>>,
				  ExplanationOnlySurface<0>,
				  ExplanationOnlySurface<1>>);
} // namespace sotoba::icp_resource::normal_known_icp_impl

namespace sotoba::icp_resource {
	using normal_known_icp_impl::NormalKnownNonSyclResource;
}

#ifdef sotoba_ENABLE_TESTING

#endif