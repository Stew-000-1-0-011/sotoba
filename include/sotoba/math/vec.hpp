#pragma once

#include <format>
#include <type_traits>
#include <utility>

#include "sotoba/use_sycl.hpp"

#include "epsilon.hpp"
#include "scalar_functions.hpp"
#include "square_mat.hpp"
#include "sym_mat.hpp"
#include "vec_forward_decl.hpp"

#ifndef sotoba_USE_SYCL
namespace sotoba::math::vec_impl {
	template<u8 n_, bool is_unit_>
	struct Vec final {
		float v[n_];

		constexpr Vec() noexcept
		{
			for(u8 i = 0; i < n_; ++i) {
				this->v[i] = 0.f;
			}
		}

		constexpr Vec(const auto ... v_) noexcept
		requires (std::same_as<std::remove_const_t<decltype(v_)>, float> && ...) && (sizeof...(v_) == n_) {
			[&]<usize ... idxs_>(std::index_sequence<idxs_ ...>) {
				((this->v[idxs_] = v_), ...);
			}(std::make_index_sequence<sizeof...(v_)>{});
		}

		template<u8 ... ns_, bool ... is_units_>
		constexpr Vec(const Vec<ns_, is_units_>& ... args) noexcept
		requires ((0 + ... + ns_) == n_) {
			u8 i = 0;
			([&]<u8 n2_, bool is_unit2_>(const Vec<n2_, is_unit2_>& arg) {
				for(u8 j = 0; j < n2_; ++j) {
					this->v[i + j] = arg.v[j];
				}
				i += n2_;
			}(args), ...);
		}

		constexpr Vec(const Vec&) noexcept = default;
		constexpr Vec(Vec&&) noexcept = default;
		constexpr auto operator=(const Vec&) noexcept -> Vec& = default;
		constexpr auto operator=(Vec&&) noexcept -> Vec& = default;
		constexpr ~Vec() noexcept = default;

		constexpr operator Vec<n_, false>() const noexcept
		requires is_unit_ {
			return [&]<usize ... idxs_>(std::index_sequence<idxs_ ...>) {
				return Vec<n_, false>(this->v[idxs_] ...);
			}(std::make_index_sequence<n_>{});
		}

		constexpr auto& x(this auto&& self) noexcept
		requires (n_ >= 1) {
			return self.v[0];
		}
		constexpr auto& y(this auto&& self) noexcept
		requires (n_ >= 2) {
			return self.v[1];
		}
		constexpr auto& z(this auto&& self) noexcept
		requires (n_ >= 3) {
			return self.v[2];
		}
		constexpr auto& w(this auto&& self) noexcept
		requires (n_ >= 4) {
			return self.v[3];
		}

		constexpr auto xyz() const noexcept -> Vec<3, false>
		requires (n_ >= 4) {
			return Vec<3, false>{this->v[0], this->v[1], this->v[2]};
		}

		constexpr auto operator-() const noexcept -> Vec {
			return *this * -1.;
		}

		constexpr auto operator+=(const Vec& rhs) noexcept -> Vec&
		requires (!is_unit_) {
			for(u8 i = 0; i < n_; ++i) {
				this->v[i] += rhs.v[i];
			}

			return *this;
		}

		constexpr auto operator-=(const Vec& rhs) noexcept -> Vec&
		requires (!is_unit_) {
			for(u8 i = 0; i < n_; ++i) {
				this->v[i] -= rhs.v[i];
			}

			return *this;
		}

		constexpr auto operator*=(const float rhs) noexcept -> Vec&
		requires (!is_unit_) {
			for(u8 i = 0; i < n_; ++i) {
				this->v[i] *= rhs;
			}

			return *this;
		}

		constexpr auto operator/=(const float rhs) noexcept -> Vec&
		requires (!is_unit_) {
			for(u8 i = 0; i < n_; ++i) {
				this->v[i] /= rhs;
			}

			return *this;
		}

		constexpr friend auto operator+(const Vec& lhs, const Vec& rhs) noexcept -> Vec<n_, false> {
			Vec<n_, false> ret = lhs;
			ret += rhs;
			return ret;
		}

		constexpr friend auto operator-(const Vec& lhs, const Vec& rhs) noexcept -> Vec<n_, false> {
			Vec<n_, false> ret = lhs;
			ret -= rhs;
			return ret;
		}

		constexpr friend auto operator*(const float lhs, const Vec& rhs) noexcept -> Vec<n_, false> {
			Vec<n_, false> ret = rhs;
			ret *= lhs;
			return ret;
		}

		constexpr friend auto operator*(const Vec& lhs, const float rhs) noexcept -> Vec<n_, false> {
			Vec<n_, false> ret = lhs;
			ret *= rhs;
			return ret;
		}

		constexpr friend auto operator/(const Vec& lhs, const float rhs) noexcept -> Vec<n_, false> {
			Vec<n_, false> ret = lhs;
			ret /= rhs;
			return ret;
		}

		constexpr decltype(auto) operator[](this auto&& self, const u8 idx) noexcept
		requires(!is_unit_) {
			return self.v[idx];
		}

		constexpr decltype(auto) operator[](const u8 idx) const noexcept {
			return this->v[idx];
		}
	};
	Vec(const float) -> Vec<1, false>;
	template<u8 ... ns_, bool ... is_units_>
	Vec(const Vec<ns_, is_units_>& ... args) -> Vec<(0 + ... + ns_)>;

	namespace vec_functions {
		template<u8 n_, bool is_unit1_, bool is_unit2_>
		inline constexpr auto dot(const Vec<n_, is_unit1_>& lhs, const Vec<n_, is_unit2_>& rhs) noexcept -> float {
			float ret = 0.f;
			for(u8 i = 0; i < n_; ++i) {
				ret += lhs.v[i] * rhs.v[i];
			}

			return ret;
		}

		template<bool is_unit1_, bool is_unit2_>
		inline constexpr auto cross(const Vec<3, is_unit1_>& lhs, const Vec<3, is_unit2_>& rhs) noexcept -> Vec<3, false> {
			return Vec3 {
				lhs.y() * rhs.z() - lhs.z() * rhs.y()
				, lhs.z() * rhs.x() - lhs.x() * rhs.z()
				, lhs.x() * rhs.y() - lhs.y() * rhs.x()
			};
		}

		template<u8 n_, bool is_unit_>
		inline constexpr auto as_uvec(const Vec<n_, is_unit_>& v) noexcept -> Vec<n_> {
			return v;
		}

		template<u8 n_, bool is_unit1_, bool is_unit2_>
		inline constexpr auto dyad(const Vec<n_, is_unit1_>& lhs, const Vec<n_, is_unit2_>& rhs) noexcept -> SquareMat<n_> {
			SquareMat<n_> ret{};
			for(u8 i = 0; i < n_; ++i) for(u8 j = 0; j < n_; ++j) {
				ret[i, j] = lhs[i] * rhs[j];
			}

			return ret;
		}

		template<u8 n_, bool is_unit_>
		inline constexpr auto self_dyad(const Vec<n_, is_unit_>& v) noexcept -> SymMat<n_> {
			SymMat<n_> ret{};
			u8 k = 0;
			for(u8 i = 0; i < n_; ++i) for(u8 j = i; j < n_; ++j) {
				ret.v[k++] = v.v[i] * v.v[j];
			}

			return ret;
		}

		template<u8 n_, bool is_unit_>
		inline constexpr auto fast_length(const Vec<n_, is_unit_>& v) noexcept -> float {
			return 1. / fast_invsqrt(dot(v, v));
		}

		template<u8 n_, bool is_unit_>
		inline constexpr auto fast_normalize(const Vec<n_, is_unit_>& v) noexcept -> Vec<n_, true> {
			const auto len = fast_length(v);
			if(len < epsilon) {
				return Vec<n_, true>{};
			}
			return as_uvec(v / len);
		}

		template<u8 n_, bool is_unit1_, bool is_unit2_>
		inline constexpr auto distance2(const Vec<n_, is_unit1_>& lhs, const Vec<n_, is_unit2_>& rhs) noexcept -> float {
			const auto diff = lhs - rhs;
			return dot(diff, diff);
		}

		template<u8 begin_, u8 end_, u8 n_, bool is_unit_>
		requires (begin_ < end_)
		inline constexpr auto split(const Vec<n_, is_unit_>& v) noexcept -> Vec<end_ - begin_, false> {
			Vec<end_ - begin_, false> ret;
			for(u8 i = begin_; i < end_; ++i) ret[i - begin_] = v[i];
			return ret;
		}

		template<u8 n_, bool is_unit_>
		inline constexpr auto diagonal(const Vec<n_, is_unit_>& diag) noexcept -> SquareMat<n_> {
			SquareMat<n_> ret{};
			for(u8 i = 0; i < n_; ++i) ret[i, i] = diag[i];
			return ret;
		}

		template<u8 n_, bool is_unit_>
		inline constexpr auto diagonal_sym(const Vec<n_, is_unit_>& diag) noexcept -> SymMat<n_> {
			SymMat<n_> ret{};
			u8 k = 0;
			for(u8 i = 0; i < n_; ++i) {
				ret.v[k] = diag[i];
				k += (n_ - i);
			}
			return ret;
		}

		template<u8 n_, bool is_init_>
		inline constexpr auto isfinite(const Vec<n_, is_init_>& v) noexcept -> bool {
			for(u8 i = 0; i < n_; ++i) if(!math::isfinite(v[i])) return false;
			return true;
		}
	}
}
#endif

namespace sotoba {
	#ifndef sotoba_USE_SYCL
	template<u8 n_, bool is_unit_>
	struct Repr<math::Vec<n_, is_unit_>> final {
		static auto repr(const math::Vec<n_, is_unit_>& self) noexcept -> std::string {
			std::string ret = std::format("Vec<{}, {}>{{", n_, is_unit_);
			for(u8 i = 0; i < n_; ++i) ret += std::format("{}, ", self[i]);
			ret += '}';
			return ret;
		}
	};
	#else
	template<int n_>
	struct Repr<math::Vec<n_>> final {
		static auto repr(const math::Vec<n_>& self) noexcept -> std::string {
			std::string ret = std::format("Vec<{}>{{", n_);
			for(int i = 0; i < n_; ++i) ret += std::format("{}, ", self[i]);
			ret += '}';
			return ret;
		}
	};
	#endif
}

namespace sotoba::math {
	#ifndef sotoba_USE_SYCL
	namespace vec = vec_impl::vec_functions;
	#else
	namespace vec {
		using sycl::cross;
		using sycl::dot;
		using sycl::fast_length;
		using sycl::fast_normalize;

		template<int n_>
		inline constexpr auto as_uvec(const Vec<n_>& v) noexcept -> Vec<n_> {
			return v;
		}

		template<int n_>
		inline constexpr auto dyad(const Vec<n_>& lhs, const Vec<n_>& rhs) noexcept -> SquareMat<n_> {
			SquareMat<n_> ret{};
			for(u8 i = 0; i < n_; ++i) for(u8 j = 0; j < n_; ++j) {
				ret[i, j] = lhs[i] * rhs[j];
			}

			return ret;
		}

		template<int n_>
		inline constexpr auto self_dyad(const Vec<n_>& v) noexcept -> SymMat<n_> {
			SymMat<n_> ret{};
			u8 k = 0;
			for(u8 i = 0; i < n_; ++i) for(u8 j = i; j < n_; ++j) {
				ret.v[k++] = v[i] * v[j];
			}

			return ret;
		}

		template<int n_>
		inline constexpr auto distance2(const Vec<n_>& lhs, const Vec<n_>& rhs) noexcept -> float {
			const auto diff = lhs - rhs;
			return dot(diff, diff);
		}

		template<int begin_, int end_, int n_>
		requires (begin_ < end_)
		inline constexpr auto split(const Vec<n_>& v) noexcept -> Vec<end_ - begin_> {
			Vec<end_ - begin_> ret;
			for(int i = begin_; i < end_; ++i) ret[i - begin_] = v[i];
			return ret;
		}

		template<int n_>
		inline constexpr auto diagonal(const Vec<n_>& diag) noexcept -> SquareMat<n_> {
			SquareMat<n_> ret{};
			for(u8 i = 0; i < n_; ++i) ret[i, i] = diag[i];
			return ret;
		}

		template<int n_>
		inline constexpr auto diagonal_sym(const Vec<n_>& diag) noexcept -> SymMat<n_> {
			SymMat<n_> ret{};
			u8 k = 0;
			for(u8 i = 0; i < n_; ++i) {
				ret.v[k] = diag[i];
				k += (n_ - i);
			}
			return ret;
		}

		template<int n_>
		inline constexpr auto isfinite(const Vec<n_>& v) noexcept -> bool {
			for(u8 i = 0; i < n_; ++i) if(!math::isfinite(v[i]))return false;
			return true;
		}
	}
	#endif
}


#ifdef sotoba_ENABLE_TESTING
#include <cmath>

#include <doctest.h>

#include "sotoba/math/approx_check.hpp"

namespace sotoba::math {
	// doctest で Vec を比較するためのヘルパー関数
	#ifndef sotoba_USE_SYCL
	template<u8 n_, bool is_unit_>
	struct ApproxCheckImpl<Vec<n_, is_unit_>> final {
		static auto compare(const Vec<n_, is_unit_>& v1, const Vec<n_, is_unit_>& v2, const std::optional<float> eps) -> bool
	#else
	template<int n_>
	struct ApproxCheckImpl<Vec<n_>> final {
		static auto compare(const Vec<n_>& v1, const Vec<n_>& v2, const std::optional<float> eps) -> bool
	#endif
		{
			for(u8 i = 0; i < n_; ++i) {
				if(v1[i] != (eps ? doctest::Approx(v2[i]).epsilon(*eps) : doctest::Approx(v2[i]))) return false;
			}
			return true;
		}
	};
}

TEST_SUITE("vec.hpp") {
	using namespace sotoba::math;

	#ifndef sotoba_USE_SYCL
	TEST_CASE("Vec Constructors") {

		using Vec1 = Vec<1, false>;
		using Vec2 = Vec<2, false>;

		SUBCASE("Default constructor") {
			constexpr Vec3 v;
			CHECK(v.x() == 0.f);
			CHECK(v.y() == 0.f);
			CHECK(v.z() == 0.f);

			constexpr Vec1 v1;
			CHECK(v1.x() == 0.f);
		}

		SUBCASE("Array constructor") {
			constexpr Vec3 v{1.f, 2.f, 3.f};
			CHECK(v.x() == 1.f);
			CHECK(v.y() == 2.f);
			CHECK(v.z() == 3.f);
		}

		SUBCASE("Scalar constructor (n_ == 1)") {
			constexpr Vec1 v(5.f);
			CHECK(v.x() == 5.f);
			
			// ユーザー定義リテラル経由
			constexpr Vec v_deduc(10.f); // Vec<1, false>
			CHECK(v_deduc.x() == 10.f);
		}

		SUBCASE("Concatenation constructor") {
			constexpr Vec1 v1(1.f);
			constexpr Vec2 v2({2.f, 3.f});
			constexpr Vec1 v3(4.f);
			
			constexpr Vec4 v4(v1, v2, v3);
			CHECK(v4.x() == 1.f);
			CHECK(v4.y() == 2.f);
			CHECK(v4.z() == 3.f);
			CHECK(v4.w() == 4.f);
		}

		SUBCASE("Copy constructor") {
			constexpr Vec3 v1({1.f, 2.f, 3.f});
			constexpr Vec3 v2 = v1;
			CHECK(v2.x() == 1.f);
			CHECK(v2.y() == 2.f);
			CHECK(v2.z() == 3.f);
		}
	}

	TEST_CASE("Vec Member Access and Conversions") {		
		SUBCASE("x, y, z, w accessors") {
			Vec4 v({1.f, 2.f, 3.f, 4.f});
			CHECK(v.x() == 1.f);
			CHECK(v.y() == 2.f);
			CHECK(v.z() == 3.f);
			CHECK(v.w() == 4.f);

			// Const access
			const Vec4 v_const = v;
			CHECK(v_const.x() == 1.f);

			// Modification
			v.x() = 10.f;
			v.y() = 20.f;
			CHECK(v.x() == 10.f);
			CHECK(v.y() == 20.f);
		}

		SUBCASE("xyz() swizzle") {
			constexpr Vec4 v({1.f, 2.f, 3.f, 4.f});
			constexpr Vec3 v_xyz = v.xyz();
			
			CHECK(v_xyz.x() == 1.f);
			CHECK(v_xyz.y() == 2.f);
			CHECK(v_xyz.z() == 3.f);
		}

		SUBCASE("as_uvec()") {
			constexpr Vec3 v({1.f, 2.f, 3.f});
			constexpr UVec3 uv = vec::as_uvec(v);
			
			// as_uvec は正規化しない（型を変えるだけ）
			CHECK(uv.x() == 1.f);
			CHECK(uv.y() == 2.f);
			CHECK(uv.z() == 3.f);
		}

		SUBCASE("UnitVec to Vec conversion") {
			constexpr UVec3 uv({1.f, 2.f, 3.f});
			constexpr Vec3 v = uv; // 暗黙の型変換
			
			CHECK(v.x() == 1.f);
			CHECK(v.y() == 2.f);
			CHECK(v.z() == 3.f);
		}
	}

	TEST_CASE("Vec Operators") {
		constexpr Vec3 v1({1.f, 2.f, 3.f});
		constexpr Vec3 v2({4.f, 5.f, 6.f});
		constexpr float s = 2.f;

		SUBCASE("Unary minus") {
			constexpr Vec3 v_neg = -v1;
			CHECK(ApproxCheck{v_neg} == ApproxCheck{Vec3{-1.f, -2.f, -3.f}});
		}

		SUBCASE("operator+=") {
			Vec3 v = v1;
			v += v2;
			CHECK(ApproxCheck{v} == ApproxCheck{Vec3{5.f, 7.f, 9.f}});
		}

		SUBCASE("operator-=") {
			Vec3 v = v2;
			v -= v1;
			CHECK(ApproxCheck{v} == ApproxCheck{Vec3{3.f, 3.f, 3.f}});
		}

		SUBCASE("operator*=") {
			Vec3 v = v1;
			v *= s;
			CHECK(ApproxCheck{v} == ApproxCheck{Vec3{2.f, 4.f, 6.f}});
		}

		SUBCASE("operator/=") {
			Vec3 v = v2;
			v /= s;
			CHECK(ApproxCheck{v} == ApproxCheck{Vec3{2.f, 2.5f, 3.f}});
		}

		SUBCASE("Binary operator+") {
			constexpr Vec3 v = v1 + v2;
			CHECK(ApproxCheck{v} == ApproxCheck{Vec3{5.f, 7.f, 9.f}});
		}

		SUBCASE("Binary operator-") {
			constexpr Vec3 v = v1 - v2;
			CHECK(ApproxCheck{v} == ApproxCheck{Vec3{-3.f, -3.f, -3.f}});
		}

		SUBCASE("Binary operator* (scalar * vec)") {
			constexpr Vec3 v = s * v1;
			CHECK(ApproxCheck{v} == ApproxCheck{Vec3{2.f, 4.f, 6.f}});
		}

		SUBCASE("Binary operator* (vec * scalar)") {
			constexpr Vec3 v = v1 * s;
			CHECK(ApproxCheck{v} == ApproxCheck{Vec3{2.f, 4.f, 6.f}});
		}

		SUBCASE("Binary operator/ (vec / scalar)") {
			constexpr Vec3 v = v2 / s;
			CHECK(ApproxCheck{v} == ApproxCheck{Vec3{2.f, 2.5f, 3.f}});
		}

		SUBCASE("self_dyad()") {
			constexpr SymMat<3> m = vec::self_dyad(v1);
			CHECK(ApproxCheck{m} == ApproxCheck{SymMat<3>{1.f, 2.f, 3.f, 4.f, 6.f, 9.f}});
		}
	}

	TEST_CASE("vec_functions") {
		constexpr Vec3 v1({1.f, 2.f, 3.f});
		constexpr Vec3 v2({4.f, 5.f, 6.f});
		constexpr UVec3 uv1 = vec::fast_normalize(v1); // 単位ベクトル

		SUBCASE("dot()") {
			// (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
			constexpr float d = vec::dot(v1, v2);
			CHECK(d == doctest::Approx(32.f));

			// 単位ベクトルとの内積
			constexpr float d_unit = vec::dot(uv1, v1);
			CHECK(d_unit == doctest::Approx(vec::fast_length(v1)));
		}

		SUBCASE("dyad()") {
			constexpr SquareMat<3> m_dyad = vec::dyad(v1, v2);
			constexpr SquareMat<3> m_expected({4.f, 5.f, 6.f, 8.f, 10.f, 12.f, 12.f, 15.f, 18.f});
			
			CHECK(ApproxCheck{m_dyad} == ApproxCheck{m_expected});
		}

		SUBCASE("fast_length()") {
			// length = sqrt(1^2 + 2^2 + 3^2) = sqrt(1 + 4 + 9) = sqrt(14)
			constexpr float len_v1 = vec::fast_length(v1);
			CHECK(len_v1 == doctest::Approx(std::sqrt(14.f)));

			constexpr float len_v2 = vec::fast_length(v2);
			// length = sqrt(4^2 + 5^2 + 6^2) = sqrt(16 + 25 + 36) = sqrt(77)
			CHECK(len_v2 == doctest::Approx(std::sqrt(77.f)));
		}

		SUBCASE("fast_normalize()") {
			constexpr Vec3 v({3.f, 0.f, 0.f});
			constexpr UVec3 uv = vec::fast_normalize(v);
			CHECK(ApproxCheck{uv} == ApproxCheck{Vec3{1.f, 0.f, 0.f}});

			constexpr Vec3 v_len = vec::fast_normalize(v1);
			constexpr float len_v1 = vec::fast_length(v1);
			CHECK(ApproxCheck{v_len} == ApproxCheck{v1 / len_v1});

			// ゼロベクトルの正規化
			constexpr Vec3 v_zero;
			constexpr UVec3 uv_zero = vec::fast_normalize(v_zero);
			CHECK(ApproxCheck{uv_zero} == ApproxCheck{Vec3{0.f, 0.f, 0.f}}); // ゼロベクトルが返る
		}
	}
	#endif

	TEST_CASE("sotoba-used") {
		constexpr Vec3 v1({1.f, 2.f, 3.f});
		constexpr Vec3 v2({4.f, 5.f, 6.f});
		
		SUBCASE("dyad()") {
			const SquareMat<3> m_dyad = vec::dyad(v1, v2);
			constexpr SquareMat<3> m_expected({4.f, 5.f, 6.f, 8.f, 10.f, 12.f, 12.f, 15.f, 18.f});
			
			CHECK(ApproxCheck{m_dyad} == ApproxCheck{m_expected});
		}

		SUBCASE("fast_length()") {
			// length = sqrt(1^2 + 2^2 + 3^2) = sqrt(1 + 4 + 9) = sqrt(14)
			const float len_v1 = vec::fast_length(v1);
			CHECK(len_v1 == doctest::Approx(std::sqrt(14.f)));

			const float len_v2 = vec::fast_length(v2);
			// length = sqrt(4^2 + 5^2 + 6^2) = sqrt(16 + 25 + 36) = sqrt(77)
			CHECK(len_v2 == doctest::Approx(std::sqrt(77.f)));
		}
	}
}
#endif