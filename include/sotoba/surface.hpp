#pragma once

#include <concepts>
#include <utility>

#include "fundamental.hpp"

namespace sotoba {
	// 性能測る前にやることか？
	// template<class T_, class U_>
	// concept with_rest_pair = requires(const T_ x) {
	// 	{[]<class Rest_>(const std::pair<U_, Rest_>&){}(x)};
	// };

	// フィッティングする表面
	template <class T_>
	concept surface = requires(const T_ imut, T_ mut, Vec4 p, const SE3 h, const Ray ray) {
		// .xyz = ローカル座標における最近接点, .w = 距離の2乗
		{ imut.closest_point_and_distance(p) } noexcept -> std::same_as<Vec4>;
		{ mut.apply_homogeneous(h) } noexcept;
		// .xyz = ローカル座標における最近接点, .w = 距離の2乗
		{ imut.ray_collision(ray) } noexcept -> std::same_as<Vec4>;
	};
} // namespace sotoba
