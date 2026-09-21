#pragma once

#include <cmath>
#include <algorithm>
#include <bit>

#include "sotoba/stdtypes.hpp"


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
		return std::clamp(x, mi, ma);
	}

	inline constexpr auto fabs(const float x) noexcept -> float {
		return std::fabs(x);
	}

	inline constexpr auto pow2(const float x) noexcept -> float {
		return x * x;
	}

	inline constexpr auto sqrt(const float x) noexcept -> float {
		return std::sqrt(x);
	}

	inline constexpr auto sin(const float x) noexcept -> float {
		return std::sin(x);
	}

	inline constexpr auto cos(const float x) noexcept -> float {
		return std::cos(x);
	}

	inline constexpr auto fmod(const float x, const float y) noexcept -> float {
		return std::fmod(x, y);
	}

	inline constexpr auto isfinite(const float x) noexcept -> bool {
		return std::isfinite(x);
	}
} // namespace sotoba::math