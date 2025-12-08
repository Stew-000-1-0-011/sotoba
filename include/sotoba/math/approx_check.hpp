#pragma once

#include <algorithm>
#include <concepts>
#include <optional>
#include <ostream>

#include "sotoba/repr.hpp"

namespace sotoba::math {
	template<class T_>
	struct ApproxCheckImpl;

	template<class T_>
	concept approx_checkable = requires(const T_ x, const T_ y, const std::optional<float> eps) {
		requires reprable<T_>;
		{ApproxCheckImpl<T_>::compare(x, y, eps)} -> std::convertible_to<bool>;
	};

	template<class T_>
	struct ApproxCheck final {
		T_ x;
		std::optional<float> eps{};

		template<class U_>
		friend auto operator==(const ApproxCheck& lhs, const ApproxCheck<U_>& rhs) -> bool {
			const auto eps = lhs.eps ? rhs.eps ? std::min(lhs.eps, rhs.eps) : lhs.eps : rhs.eps;
			return ApproxCheckImpl<T_>::compare(lhs.x, rhs.x, eps);
		}

		friend auto operator<<(std::ostream& os, const ApproxCheck& self) -> std::ostream& {
			return os << Repr<T_>::repr(self.x);
		}
	};
}
