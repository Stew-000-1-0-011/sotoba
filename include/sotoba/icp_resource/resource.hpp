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

	template <class T_, class... Ss_>
	concept icp_resource = (surfacelike<Ss_> && ...)
		&& requires(T_ mut,
					const T_ imut,
					std::tuple<std::vector<Ss_>...> surfs,
					std::array<std::vector<ObjSurfId>, 2> osids,
					u8 obj_num,
					u8 points_num,
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
	using resource_impl::to_resource;
} // namespace sotoba::icp_resource