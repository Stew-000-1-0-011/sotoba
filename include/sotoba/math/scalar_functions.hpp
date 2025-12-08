#pragma once

#include <cmath>
#include <algorithm>
#include <bit>

#include "sotoba/stdtypes.hpp"

#ifdef sotoba_USE_SYCL
#include "sotoba/use_sycl.hpp"
#endif

namespace sotoba::math {
	inline constexpr auto fast_invsqrt(const float x) noexcept -> float {
		static_assert(
			sizeof(float) == sizeof(i32),
			"float must be 32 bits (IEEE 754 single precision)"
		);
		
		const float xhalf = 0.5f * x;
		i32 i = std::bit_cast<i32>(x);
		i = 0x5F3759DF - (i >> 1);
		float y = std::bit_cast<float>(i);
		y = y * (1.5f - (xhalf * y * y));
		y = y * (1.5f - (xhalf * y * y));

		return y;
	}

	inline constexpr auto clamp(const float x, const float mi, const float ma) noexcept -> float {
		#ifdef sotoba_USE_SYCL
		return sycl::clamp(x, mi, ma);
		#else
		return std::clamp(x, mi, ma);
		#endif
	}

	inline constexpr auto fabs(const float x) noexcept -> float {
		#ifdef sotoba_USE_SYCL
		return sycl::fabs(x);
		#else
		return std::fabs(x);
		#endif
	}

	inline constexpr auto pow2(const float x) noexcept -> float {
		return x * x;
	}

	inline constexpr auto sqrt(const float x) noexcept -> float {
		#ifdef sotoba_USE_SYCL
		return sycl::sqrt(x);
		#else
		return std::sqrt(x);
		#endif
	}

	inline constexpr auto sin(const float x) noexcept -> float {
		#ifdef sotoba_USE_SYCL
		return sycl::sin(x);
		#else
		return std::sin(x);
		#endif
	}

	inline constexpr auto cos(const float x) noexcept -> float {
		#ifdef sotoba_USE_SYCL
		return sycl::cos(x);
		#else
		return std::cos(x);
		#endif
	}

	inline constexpr auto fmod(const float x, const float y) noexcept -> float {
		#ifdef sotoba_USE_SYCL
		return sycl::fmod(x, y);
		#else
		return std::fmod(x, y);
		#endif
	}

	inline constexpr auto isfinite(const float x) noexcept -> float {
		#ifdef sotoba_USE_SYCL
		return sycl::isfinite(x);
		#else
		return std::isfinite(x);
		#endif
	}
} // namespace sotoba::math