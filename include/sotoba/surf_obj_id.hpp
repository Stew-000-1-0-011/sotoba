#pragma once

#include <utility>
#include "stdtypes.hpp"

namespace sotoba {
	enum class ObjId : u8 {};
	enum class SurfId : u8 {};
	enum class ObjSurfId : u16 {
		Null = u16(-1)
	};

	inline auto osid_pack(const ObjId oid, const SurfId sid) noexcept -> ObjSurfId {
		return ObjSurfId(u16(sid) << 8 | u16(oid));
	}

	inline auto osid_depack(const ObjSurfId osid) noexcept -> std::pair<ObjId, SurfId> {
		return {ObjId(u16(osid) >> 8), SurfId(u16(osid) & 0xFF)};
	}
}