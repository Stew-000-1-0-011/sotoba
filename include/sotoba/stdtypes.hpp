#pragma once

#include <cstddef>
#include <cstdint>
#include <version>

// Clang 18 は P0847 を実装しているが __cpp_explicit_this_parameter を定義しない。
#if !defined(__cpp_explicit_this_parameter)
	#if defined(__clang__)
		#if __clang_major__ < 18
			#error "sotoba requires deducing this (P0847). Clang 18 or later is required."
		#endif
	#elif defined(__GNUC__)
		#if __GNUC__ < 14
			#error "sotoba requires deducing this (P0847). GCC 14 or later is required."
		#endif
	#endif
#endif

#if !defined(__cpp_multidimensional_subscript)
	#error "sotoba requires multidimensional subscript (P2128). Compile with -std=c++23 on GCC 14+ / Clang 18+."
#endif

#if !defined(__cpp_lib_format)
	#error "sotoba requires <format>. Use libstdc++ 13+ or libc++ 17+."
#endif

namespace sotoba::stdtypes {
	using u8 = std::uint8_t;
	using u16 = std::uint16_t;
	using u64 = std::uint64_t;
	using u32 = std::uint32_t;
	using i32 = std::int32_t;
	using usize = std::size_t;
} // namespace sotoba::stdtypes

namespace sotoba {
	using namespace stdtypes;
}