#pragma once

#include <format>
#include "sotoba/repr.hpp"
#include "sotoba/stdtypes.hpp"

#include "vec_forward_decl.hpp"

namespace sotoba::math {
	template <u8 d_>
	struct SquareMat final {
		float v[d_ * d_]{};

		static constexpr auto ide() noexcept -> SquareMat {
			SquareMat ret{};
			for (u8 i = 0; i < d_; ++i) ret[i, i] = 1.f;
			return ret;
		}

		constexpr decltype(auto) operator[](this auto&& self, const u8 i, const u8 j) noexcept {
			return self.v[d_ * i + j];
		}

		inline constexpr auto operator[](this auto&& self, const u8 i) noexcept -> Vec<d_, false> {
			Vec<d_, false> ret;
			for (u8 j = 0; j < d_; ++j) { ret[j] = self[i, j]; }
			return ret;
		}

		constexpr auto operator-() const noexcept -> SquareMat {
			return *this * -1.;
		}

		constexpr auto operator+=(const SquareMat& rhs) noexcept -> SquareMat& {
			for (u8 i = 0; i < d_ * d_; ++i) { this->v[i] += rhs.v[i]; }

			return *this;
		}

		constexpr auto operator-=(const SquareMat& rhs) noexcept -> SquareMat& {
			for (u8 i = 0; i < d_ * d_; ++i) { this->v[i] -= rhs.v[i]; }

			return *this;
		}

		constexpr auto operator*=(const float rhs) noexcept -> SquareMat& {
			for (u8 i = 0; i < d_ * d_; ++i) { this->v[i] *= rhs; }

			return *this;
		}

		constexpr auto operator/=(const float rhs) noexcept -> SquareMat& {
			for (u8 i = 0; i < d_ * d_; ++i) { this->v[i] /= rhs; }

			return *this;
		}

		constexpr friend auto operator+(const SquareMat& lhs, const SquareMat& rhs) noexcept
			-> SquareMat {
			SquareMat ret = lhs;
			ret += rhs;
			return ret;
		}

		constexpr friend auto operator-(const SquareMat& lhs, const SquareMat& rhs) noexcept
			-> SquareMat {
			SquareMat ret = lhs;
			ret -= rhs;
			return ret;
		}

		constexpr friend auto operator*(const float lhs, const SquareMat& rhs) noexcept
			-> SquareMat {
			SquareMat ret = rhs;
			ret *= lhs;
			return ret;
		}

		constexpr friend auto operator*(const SquareMat& lhs, const float rhs) noexcept
			-> SquareMat {
			SquareMat ret = lhs;
			ret *= rhs;
			return ret;
		}

		constexpr friend auto operator/(const SquareMat& lhs, const float rhs) noexcept
			-> SquareMat {
			SquareMat ret = lhs;
			ret /= rhs;
			return ret;
		}

		template <bool is_unit_>
		constexpr friend auto operator*(const SquareMat& lhs, const Vec<d_, is_unit_>& rhs) noexcept
			-> Vec<d_, false>
		{
			Vec<d_, false> ret{};
			for (u8 i = 0; i < d_; ++i)
				for (u8 j = 0; j < d_; ++j) { ret[i] += lhs[i, j] * rhs[j]; }
			return ret;
		}

		constexpr auto transpose() const noexcept -> SquareMat {
			SquareMat ret = *this;
			for (u8 i = 0; i < d_; ++i)
				for (u8 j = i + 1; j < d_; ++j) {
					const auto tmp = ret[i, j];
					ret[i, j] = ret[j, i];
					ret[j, i] = tmp;
				}
			return ret;
		}
	};
} // namespace sotoba::math

namespace sotoba {
	template <u8 d_>
	struct Repr<math::SquareMat<d_>> final {
		static auto repr(const math::SquareMat<d_>& self) -> std::string {
			std::string ret = std::format("SquareMat<{}>{{", d_);
			for (u8 i = 0; i < d_; ++i) {
				ret += '{';
				for (u8 j = 0; j < d_; ++j) { ret += std::format("{}, ", self[i, j]); }
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
	// doctest で SquareMat を比較するためのヘルパー関数
	template <u8 n_>
	struct ApproxCheckImpl<SquareMat<n_>> final {
		static auto compare(
			const SquareMat<n_>& m1,
			const SquareMat<n_>& m2,
			const std::optional<float> eps = 1.e-4f
		) -> bool {
			for (u8 i = 0; i < n_; ++i)
				for (u8 j = 0; j < n_; ++j) {
					if (m1[i, j]
						!= (eps ? doctest::Approx(m2[i, j]).epsilon(*eps)
								: doctest::Approx(m2[i, j])))
						return false;
				}
			return true;
		}
	};
} // namespace sotoba::math
#endif
