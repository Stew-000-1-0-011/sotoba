#pragma once

#include <concepts>
#include <utility>

#include "sotoba/math/vec.hpp"
#include "sotoba/math/se3.hpp"

namespace sotoba::surface::surface_impl {
	using math::Vec3;
	using math::UVec3;
	using math::Vec4;
	using math::SE3;

	template<class T_>
	concept surfacelike = requires(const T_ imut, T_ mut, Vec3 p, const SE3 h, const UVec3 ray) {
		// .xyz = 最近接点, .w = 距離の二乗
		{ imut.closest_pd(p) } noexcept -> std::same_as<Vec4>;
		// .first.xyz = 最近接点, .first.w = 距離の二乗, .second = 法線
		{ imut.closest_pdn(p) } noexcept -> std::same_as<std::pair<Vec4, UVec3>>;
		// 剛体変換を加える
		{ mut.apply_se3(h) } noexcept;
		// 距離の二乗
		{ imut.ray_collision(ray) } noexcept -> std::same_as<float>;
	};

	template<int>
	struct ExplanationOnlySurface final {
		auto closest_pd(const Vec3& p) const noexcept -> Vec4;
		auto closest_pdn(const Vec3& p) const noexcept -> std::pair<Vec4, UVec3>;
		void apply_se3(const SE3& h) noexcept;
		auto ray_collision(const UVec3& ray) const noexcept -> float;
	};
	static_assert(surfacelike<ExplanationOnlySurface<0>>);
}

namespace sotoba::surface {
	using surface_impl::surfacelike;
	using surface_impl::ExplanationOnlySurface;
}