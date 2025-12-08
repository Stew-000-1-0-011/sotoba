#pragma once

#include <array>
#include <vector>

#include "sotoba/math/se3.hpp"
#include "sotoba/surf_obj_id.hpp"
#include "sotoba/surface/surface.hpp"

namespace sotoba::sim {
	template<surface::surfacelike ... Ss_>
	struct NonSyclSimResource final {
		std::tuple<std::vector<Ss_> ...> surfs;
		std::tuple<std::vector<Ss_> ...> moved_surfs;
		std::array<std::vector<ObjId>, sizeof...(Ss_)> oids;
		std::vector<math::SE3> obj_poses;

		
	};
}