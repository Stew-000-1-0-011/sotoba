#pragma once

#include <cmath>
#include <algorithm>
#include <bit>

#include "sotoba/stdtypes.hpp"

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