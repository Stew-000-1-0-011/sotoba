#pragma once

#include <concepts>
#include <string>

namespace sotoba {
	template<class T_>
	struct Repr;

	template<class T_>
	concept reprable = requires(const T_ imut) {
		{Repr<T_>::repr(imut)} -> std::convertible_to<std::string>;
	};
}
