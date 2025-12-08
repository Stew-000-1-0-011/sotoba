#pragma once

#include <cstddef>
#include <cstdint>

namespace sotoba::stdtypes {
	using u8 = std::uint8_t;
	using u16 = std::uint16_t;
	using u64 = std::uint64_t;
	using u32 = std::uint32_t;
	using i32 = std::int32_t;
	using usize = std::size_t;
}

namespace sotoba {
	using namespace stdtypes;
}