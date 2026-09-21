#pragma once

#include <concepts>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "sotoba/math/se3.hpp"
#include "sotoba/math/vec.hpp"
#include "sotoba/surf_obj_id.hpp"
#include "sotoba/surface/surface.hpp"

namespace sotoba::icp_resource::resource_impl {
	using math::SE3;
	using math::Vec3;
	using surface::ExplanationOnlySurface;
	using surface::surfacelike;

	/// run_icp の呼び出し自体の成否。
	enum class IcpError : u8 {
		none = 0,
		/// point_cloud.size() が確保済みバッファ容量 (points_capacity()) を超えている。
		/// バッファの再確保はしない。呼び出しは何も行わず、姿勢も状態も変化しない。
		too_many_points,
		/// weighting に不正な値 (負の σ、0 以下の huber_k、非有限値) が指定された。
		/// 呼び出しは何も行わず、姿勢も状態も変化しない。
		invalid_weighting,
	};

	/// オブジェクトごとの、直近の run_icp における姿勢更新の結果。
	enum class ObjStatus : u8 {
		/// まだ一度も run_icp が走っていない。
		not_run = 0,
		/// 姿勢が更新された。
		updated,
		/// 対応点が min_correspondences 未満で、姿勢は run_icp 呼び出し時の値のまま。
		too_few_correspondences,
		/// 線形方程式 (コレスキー分解) が解けず、姿勢は直前の値のまま。
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
			   { mut.obj_pose(oid) } -> std::convertible_to<SE3&>;
			   { imut.obj_pose(oid) } -> std::convertible_to<const SE3&>;
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
	using resource_impl::ObjStatus;
	using resource_impl::to_resource;
} // namespace sotoba::icp_resource