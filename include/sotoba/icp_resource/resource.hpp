#pragma once

#include <concepts>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <Eigen/Dense>

#include "sotoba/math/se3.hpp"
#include "sotoba/math/vec.hpp"
#include "sotoba/surf_obj_id.hpp"
#include "sotoba/surface/surface.hpp"

namespace sotoba::icp_resource::resource_impl {
	using math::Vec3;
	using surface::ExplanationOnlySurface;
	using surface::surfacelike;

	/// run_icp の呼び出し自体の成否。
	enum class IcpError : u8 {
		none = 0,
		/// 点数が points_capacity() を超えている。再確保はしない。
		too_many_points,
		invalid_weighting,
		invalid_accept_schedule,
		invalid_accept_distance,
		invalid_loop_num,
		prior_size_mismatch,
		/// 非ゼロの事前分布があるのに weighting.noise が無い。
		/// Λ は物理単位を持つので、A も 1/σ² の重みで組まないと足せない。
		prior_requires_noise_model,
		/// 事前残差 r = log(pose mean⁻¹) を線形化できない。
		/// Sophus の log() は回転角を [0, π] に畳むので、実際に発火するのは
		/// pose または mean が非有限な場合。
		prior_linearization_failed,
		/// 事前分布の information が非対称、または mean を含め非有限。
		invalid_prior_information,
	};

	/// 点対面残差の分散を σ_r² cos² + r² σ_θ² (1 - cos²) と見積もる誤差モデル。
	struct NoiseModel final {
		float sigma_range; ///< [m]
		float sigma_angle; ///< [rad]
	};

	struct IcpWeighting final {
		/// 無指定なら全点の重みが 1。
		std::optional<NoiseModel> noise{};
		/// 正規化残差 e/σ に対する Huber の閾値。noise 無指定なら σ = 1 なので単位は [m]。
		std::optional<float> huber_k{};
	};

	/// 事前分布。information が mean まわりの左摂動 T = exp(ξ) mean の座標での情報行列 Λ。
	/// ゼロ行列は事前なしを表す。
	struct ObjPrior final {
		Sophus::SE3f mean{};
		Eigen::Matrix<float, 6, 6> information = Eigen::Matrix<float, 6, 6>::Zero();
	};

	/// 物体自身の座標系の摂動 T = mean exp(ξ_b) で持っている情報行列 Λ_b を、
	/// ObjPrior::information が要求する左摂動 T = exp(ξ) mean の座標に直す。
	/// ξ = Ad(mean) ξ_b なので Λ = Ad(mean⁻¹)ᵀ Λ_b Ad(mean⁻¹)。
	inline auto prior_information_from_body(
		const Sophus::SE3f& mean,
		const Eigen::Matrix<float, 6, 6>& information_body
	) noexcept -> Eigen::Matrix<float, 6, 6> {
		const Eigen::Matrix<float, 6, 6> ad = mean.inverse().Adj();
		const Eigen::Matrix<float, 6, 6> ret = ad.transpose() * information_body * ad;
		return 0.5f * (ret + ret.transpose());
	}

	/// 全メンバに既定値があるが、max_loop_num と accept_distance2 は run_icp が
	/// 検証するので、既定値のまま呼ぶとエラーが返る。
	///
	/// tikhonov は Sophus::SE3f::Tangent と同じ (並進, 回転) の順で、A = Σ w JᵀJ の対角に
	/// そのまま加わる。重みの総和では割らない。
	///
	/// priors は非所有ビュー。IcpParams を保存して呼び出しをまたいで使わない。
	/// 空、または obj_num と同じ長さ。information がゼロ行列のオブジェクトは事前なしと
	/// 同じに扱われる。非ゼロの事前が1つでもあるなら weighting.noise が必須。
	struct IcpParams final {
		u32 max_loop_num = 1;
		float accept_distance2 = 0.f;
		float convergence_delta2 = 0.f;
		float accept_distance2_begin = 0.f;
		Sophus::SE3f::Tangent tikhonov = Sophus::SE3f::Tangent::Zero();
		IcpWeighting weighting{};
		std::span<const ObjPrior> priors{};
	};

	/// オブジェクトごとの、直近の run_icp における姿勢更新の結果。
	enum class ObjStatus : u8 {
		not_run = 0,
		updated,
		/// 姿勢は run_icp 呼び出し時の値のまま。
		too_few_correspondences,
		/// 姿勢は直前の値のまま。
		solve_failed,
	};

	template <class T_, class... Ss_>
	concept icp_resource = (surfacelike<Ss_> && ...)
		&& requires(T_ mut,
					const T_ imut,
					std::tuple<std::vector<Ss_>...> surfs,
					std::array<std::vector<ObjSurfId>, sizeof...(Ss_)> osids,
					u8 obj_num,
					usize points_num,
					u8 oid) {
			   { T_{std::move(surfs), std::move(osids), obj_num, points_num} };
			   { mut.obj_pose(oid) } -> std::convertible_to<Sophus::SE3f&>;
			   { imut.obj_pose(oid) } -> std::convertible_to<const Sophus::SE3f&>;
		   };

	template <template <class...> class Resource_, surfacelike... Ss_>
		requires icp_resource<Resource_<Ss_...>, Ss_...>
	inline auto to_resource(
		const usize points_num,
		std::span<const std::span<const std::variant<Ss_...>>>&& objects
	) -> Resource_<Ss_...> {
		if (objects.size() > 255) throw std::runtime_error{"too much objects."};
		const u8 obj_num = objects.size();

		std::tuple<std::vector<Ss_>...> surfs{};
		std::array<std::vector<ObjSurfId>, sizeof...(Ss_)> osids{};
		SurfId next{0};
		for (u8 iobj = 0; iobj < obj_num; ++iobj) {
			const auto obj = objects[iobj];

			for (const auto& surface_variant : obj) {
				[&]<usize... idxs_>(std::index_sequence<idxs_...>) {
					(
						[&]<surfacelike S_>(std::vector<S_>& surfs, std::vector<ObjSurfId>& osids) {
							if (const S_ * const p = std::get_if<S_>(&surface_variant)) {
								const S_& s = *p;

								if (next == static_cast<SurfId>(0xFF)) {
									throw std::runtime_error{"too much surface"};
								} else {
									surfs.emplace_back(s);
									osids.emplace_back(osid_pack(ObjId(iobj), next));
									next = static_cast<SurfId>(std::to_underlying(next) + 1);
								}
							}
						}(std::get<idxs_>(surfs), osids[idxs_]),
						...
					);
				}(std::index_sequence_for<Ss_...>{});
			}
		}

		return Resource_<Ss_...>{std::move(surfs), std::move(osids), obj_num, points_num};
	}
} // namespace sotoba::icp_resource::resource_impl

namespace sotoba::icp_resource {
	using resource_impl::icp_resource;
	using resource_impl::IcpError;
	using resource_impl::IcpParams;
	using resource_impl::IcpWeighting;
	using resource_impl::NoiseModel;
	using resource_impl::ObjPrior;
	using resource_impl::ObjStatus;
	using resource_impl::prior_information_from_body;
	using resource_impl::to_resource;
} // namespace sotoba::icp_resource