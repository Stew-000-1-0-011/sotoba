#pragma once

#include <stdexcept>
#include <string_view>

namespace sotoba {
	struct LogicException: std::logic_error {
		constexpr LogicException(const std::string_view msg) noexcept
			: std::logic_error{std::string{msg}} {}
	};
} // namespace sotoba