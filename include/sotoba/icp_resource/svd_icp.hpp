#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "sotoba/stdtypes.hpp"
#include "sotoba/surf_obj_id.hpp"

#include "sotoba/math/quaternion.hpp"
#include "sotoba/math/se3.hpp"
#include "sotoba/math/square_mat.hpp"
#include "sotoba/math/vec.hpp"
#include "sotoba/surface/surface.hpp"

#include "resource.hpp"


namespace sotoba::icp_resource::svd_icp_impl {
	using Eigen::Matrix3f;

	using math::SE3;
	using math::SquareMat;
	using math::Vec;
	using math::Vec3;
	using math::Vec4;
	namespace vec = math::vec;
	using surface::surfacelike;
	using surface::ExplanationOnlySurface;

	template<surfacelike ... Surfaces_>
	struct SvdNonSyclResource final {
		// 表面とその情報、座標変換後の表面のバッファ
		std::tuple<std::vector<Surfaces_> ...> surfs;
		std::tuple<std::vector<Surfaces_> ...> moved_surfs;
		std::array<std::vector<ObjSurfId>, sizeof...(Surfaces_)> osids;

		// 点群とその最近接点に関する情報
		std::vector<std::pair<Vec4, ObjSurfId>> qs;

		// 加算されていくやつら
		std::vector<Vec3> p_centroids;
		std::vector<Vec3> q_centroids;
		std::vector<usize> counts;
		std::vector<SquareMat<3>> covariance_matrixs;

		// ここに入れた姿勢をもとに、ICPがはしり、補正された結果がここに入る
		std::vector<SE3> obj_poses;

		u8 obj_num;

		SvdNonSyclResource (
			std::tuple<std::vector<Surfaces_> ...>&& surfs
			, std::array<std::vector<ObjSurfId>, sizeof...(Surfaces_)>&& osids
			, const u8 obj_num
			, const usize points_num
		) noexcept
		: surfs{std::move(surfs)}
		, moved_surfs{}
		, osids{std::move(osids)}
		, qs{}
		, p_centroids{}
		, q_centroids{}
		, counts{}
		, covariance_matrixs{}
		, obj_poses{}
		, obj_num{obj_num}
		{
			([&]<surfacelike S_>() {
				std::get<std::vector<S_>>(this->moved_surfs) = std::get<std::vector<S_>>(this->surfs);
			}.template operator()<Surfaces_>(), ...);
			this->qs.resize(points_num);
			this->p_centroids.resize(obj_num);
			this->q_centroids.resize(obj_num);
			this->counts.resize(obj_num);
			this->covariance_matrixs.resize(obj_num);
			this->obj_poses.resize(obj_num, SE3::ide());
		}

		decltype(auto) obj_pose(this auto&& self, const u8 oid) noexcept {
			return self.obj_poses[oid];
		}

		void run_icp (
			std::vector<Vec3>&& point_cloud
			, const Vec3& tikhonov
			, const u32 loop_num
			, const float accept_distance2
		) noexcept {
			
			for(u32 iloop = 0; iloop < loop_num; ++iloop) {
				// surfsをobj_posesに従い移動
				[&]<usize ... idxs_>(std::index_sequence<idxs_ ...>) {
					([&]<surfacelike S_>(const std::vector<S_>& surf, const std::vector<ObjSurfId>& osid, std::vector<S_>& moved_surf) {
						for(usize i = 0; i < surf.size(); ++i) {
							const auto [sid, oid] = osid_depack(osid[i]);
							moved_surf[i] = surf[i];
							moved_surf[i].apply_se3(this->obj_poses[u8(oid)]);
						}
					}(std::get<idxs_>(this->surfs), this->osids[idxs_], std::get<idxs_>(this->moved_surfs)), ...);
				}(std::index_sequence_for<Surfaces_ ...>{});

				// 各点の最近接点をqsに格納
				[&]<usize ... idxs_>(std::index_sequence<idxs_ ...>) {
					for(usize ip = 0; ip < point_cloud.size(); ++ip) {
						if(!vec::isfinite(point_cloud[ip])) {
							this->qs[ip] = {{Vec3{}, Vec{std::numeric_limits<float>::infinity()}}, ObjSurfId::Null};
						}

						std::pair<Vec4, ObjSurfId> q{{Vec3{}, Vec{std::numeric_limits<float>::infinity()}}, ObjSurfId::Null};
						([&]<surfacelike S_>(const std::vector<S_>& surf, const u8 isurf_kind) {
							for(u8 isurf = 0; isurf < surf.size(); ++isurf) {
								const auto q_ = surf[isurf].closest_pd(point_cloud[ip]);
								if(q_.w() < q.first.w()) {
									q = {q_, this->osids[isurf_kind][isurf]};
								}
							}
						}(std::get<idxs_>(this->moved_surfs), idxs_), ...);

						this->qs[ip] = q;
					}
				}(std::index_sequence_for<Surfaces_ ...>{});

				// 重心計算
				for(u8 iobj = 0; iobj < this->obj_num; ++iobj) {
					this->p_centroids[iobj] = Vec3{};
					this->q_centroids[iobj] = Vec3{};
					this->counts[iobj] = 0;
				}
				for(usize ip = 0; ip < point_cloud.size(); ++ip) {
					const auto [qd, osid] = this->qs[ip];
					if(accept_distance2 < qd.w()) {
						continue;
					}
					const u8 iobj = std::to_underlying(osid_depack(osid).second);
					this->p_centroids[iobj] += point_cloud[ip];
					this->q_centroids[iobj] += qd.xyz();
					this->counts[iobj]++;
				}
				for(u8 iobj = 0; iobj < this->obj_num; ++iobj) {
					if(this->counts[iobj] < 3) continue;  // 点が少なすぎるオブジェクトはスキップ
					this->p_centroids[iobj] /= this->counts[iobj];
					this->q_centroids[iobj] /= this->counts[iobj];
					this->covariance_matrixs[iobj] = SquareMat<3>{};
				}
				for(usize ip = 0; ip < point_cloud.size(); ++ip) {
					const auto [qd, osid] = this->qs[ip];
					if(accept_distance2 < qd.w()) {
						continue;
					}
					const u8 iobj = std::to_underlying(osid_depack(osid).second);
					// qをpに近づける
					this->covariance_matrixs[iobj] += vec::dyad(Vec3{qd.xyz() - q_centroids[iobj]}, point_cloud[ip] - p_centroids[iobj]);
				}

				const auto tikhonov_ = vec::diagonal(tikhonov);
				// 鏡映に注意してSVD(回転行列についてtikkonov正則化を加える = 共分散行列にdiagonal(tikhonov)を加える)
				for(u8 iobj = 0; iobj < this->obj_num; ++iobj) {
					if(this->counts[iobj] < 3) continue;  // 点が少なすぎるオブジェクトはスキップ

					this->covariance_matrixs[iobj] += tikhonov_;
					Matrix3f cov{};
					for(u8 i = 0; i < 3; ++i) for(u8 j = 0; j < 3; ++j) {
						cov(i, j) = this->covariance_matrixs[iobj][i, j];
					}
					
					Eigen::JacobiSVD<Matrix3f> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
					const Matrix3f u = svd.matrixU();
					const Matrix3f v = svd.matrixV();

					// R' = V * U^T
					const Matrix3f r = v * u.transpose();

					Eigen::Quaternionf rot{};
					// 鏡映チェックと補正
					if (r.determinant() < 0.f) {
						// 行列式が負の場合、鏡映が発生しているため補正が必要
						// S行列を構築し、Vの第3列を反転させる
						Matrix3f s = Matrix3f::Identity();
						s(2, 2) = -1.f;
						
						// 補正された回転行列 R の計算
						// R = V * S * U^T
						const Matrix3f r = v * s * u.transpose();
						rot = Eigen::Quaternionf{r};
					}
					else {
						rot = Eigen::Quaternionf{r};
					}

					const auto qua = math::Quaternion{Vec4{rot.x(), rot.y(), rot.z(), rot.w()}}.normalize();
					const auto t = p_centroids[iobj] - rot_vec(qua, q_centroids[iobj]);

					// 姿勢を更新
					this->obj_poses[iobj] = (SE3{qua, t} * this->obj_poses[iobj]).normalize();
				}
			}
		}
	};
	static_assert(icp_resource::icp_resource <
		SvdNonSyclResource<ExplanationOnlySurface<0>, ExplanationOnlySurface<1>>
		, ExplanationOnlySurface<0>
		, ExplanationOnlySurface<1>
	>);
}

namespace sotoba::icp_resource {
	using svd_icp_impl::SvdNonSyclResource;
}

#ifdef sotoba_ENABLE_TESTING

#endif