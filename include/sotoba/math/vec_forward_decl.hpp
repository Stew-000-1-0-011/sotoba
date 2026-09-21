#pragma once

#include "sotoba/stdtypes.hpp"

namespace sotoba::math {
	namespace vec_impl {
		template <u8 n_, bool is_unit_ = false>
		struct Vec;
	}

	using vec_impl::Vec;
	using Vec3 = Vec<3>;
	using Vec4 = Vec<4>;
	using Vec6 = Vec<6>;
	template <u8 n_>
	using UVec = Vec<n_, true>;
	using UVec3 = Vec<3, true>;
	using UVec4 = Vec<4, true>;
} // namespace sotoba::math
