#pragma once

#include <ostream>

#include "sotoba/repr.hpp"
#include "sotoba/stdtypes.hpp"

namespace sotoba::math {
	// 上三角でなく、下三角で持つべきだった(添え字計算が辛い)
	template <u8 d_>
	struct SymMat final {
		static constexpr u16 len = d_ * (d_ + 1) / 2;
		float v[len]{};

		static constexpr auto ide() noexcept -> SymMat {
			SymMat ret{};
			u8 k = 0;
			for (u8 i = 0; i < d_; ++i) {
				ret.v[k] = 1.;
				k += (d_ - i);
			}
			return ret;
		}

		constexpr decltype(auto) operator[](this auto&& self, u8 i, u8 j) noexcept {
			if (i > j) std::swap(i, j);
			return self.v[i * d_ - i * (i + 1) / 2 + j];
		}

		constexpr auto operator-() const noexcept -> SymMat {
			return *this * -1.;
		}

		constexpr auto operator+=(const SymMat& rhs) noexcept -> SymMat& {
			for (u16 i = 0; i < len; ++i) { this->v[i] += rhs.v[i]; }

			return *this;
		}

		constexpr auto operator-=(const SymMat& rhs) noexcept -> SymMat& {
			for (u16 i = 0; i < len; ++i) { this->v[i] -= rhs.v[i]; }

			return *this;
		}

		constexpr auto operator*=(const float rhs) noexcept -> SymMat& {
			for (u16 i = 0; i < len; ++i) { this->v[i] *= rhs; }

			return *this;
		}

		constexpr auto operator/=(const float rhs) noexcept -> SymMat& {
			for (u16 i = 0; i < len; ++i) { this->v[i] /= rhs; }

			return *this;
		}

		constexpr friend auto operator+(const SymMat& lhs, const SymMat& rhs) noexcept -> SymMat {
			SymMat ret = lhs;
			ret += rhs;
			return ret;
		}

		constexpr friend auto operator-(const SymMat& lhs, const SymMat& rhs) noexcept -> SymMat {
			SymMat ret = lhs;
			ret -= rhs;
			return ret;
		}

		constexpr friend auto operator*(const float lhs, const SymMat& rhs) noexcept -> SymMat {
			SymMat ret = rhs;
			ret *= lhs;
			return ret;
		}

		constexpr friend auto operator*(const SymMat& lhs, const float rhs) noexcept -> SymMat {
			SymMat ret = lhs;
			ret *= rhs;
			return ret;
		}

		constexpr friend auto operator/(const float lhs, const SymMat& rhs) noexcept -> SymMat {
			SymMat ret = rhs;
			ret /= lhs;
			return ret;
		}

		constexpr friend auto operator/(const SymMat& lhs, const float rhs) noexcept -> SymMat {
			SymMat ret = lhs;
			ret /= rhs;
			return ret;
		}
	};
} // namespace sotoba::math

namespace sotoba {
	template <u8 d_>
	struct Repr<math::SymMat<d_>> final {
		static auto repr(const math::SymMat<d_>& self) -> std::string {
			std::string ret = std::format("SymMat<{}>{{", d_);
			u16 k = 0;
			for (u8 i = 0; i < d_; ++i) {
				ret += '{';
				for (u8 j = i; j < d_; ++j) { ret += std::format("{}, ", self.v[k++]); }
				ret += '}';
			}
			ret += '}';
			return ret;
		}
	};
} // namespace sotoba

#ifdef sotoba_ENABLE_TESTING

	#include <optional>

	#include <doctest.h>

	#include "sotoba/math/approx_check.hpp"

namespace sotoba::math {
	template <u8 n_>
	struct ApproxCheckImpl<math::SymMat<n_>> final {
		static auto
		compare(const SymMat<n_>& m1, const SymMat<n_>& m2, const std::optional<float> eps = 1.e-4f)
			-> bool {
			for (u8 i = 0; i < n_ * (n_ + 1) / 2; ++i) {
				if (m1.v[i]
					!= (eps ? doctest::Approx(m2.v[i]).epsilon(*eps) : doctest::Approx(m2.v[i])))
					return false;
			}
			return true;
		}
	};
} // namespace sotoba::math
#endif
