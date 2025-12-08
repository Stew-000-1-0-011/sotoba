#pragma once

#include "sotoba/stdtypes.hpp"
#include "sotoba/use_sycl.hpp"

namespace sotoba::math {
	#ifndef sotoba_USE_SYCL
	namespace vec_impl {
		template<u8 n_, bool is_unit_ = false>
		struct Vec;
	}
	#else
	namespace vec_impl {
		template<int n_, bool = false>
		using Vec = sycl::vec<float, n_>;
	}
	#endif

	using vec_impl::Vec;
	using Vec3 = Vec<3>;
	using Vec4 = Vec<4>;
	template<u8 n_>
	using UVec = Vec<n_, true>;
	using UVec3 = Vec<3, true>;
	using UVec4 = Vec<4, true>;
}