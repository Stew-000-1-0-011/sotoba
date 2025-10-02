#include <numbers>

#include "fundamental.hpp"

namespace sotoba {
	/**
 * @brief 2つの32ビット符号なし整数から2つの一様乱数float [0,1) を生成
 */
	inline constexpr auto uints_to_uniform_floats(const sycl::uint2 v) noexcept -> sycl::float2 {
		// 2.3283064365386963e-10f は 1.0f / (2^32)
		return {(float)v.x() * 2.3283064365386963e-10f, (float)v.y() * 2.3283064365386963e-10f};
	}

	/**
 * @brief Box-Muller変換：2つの一様乱数から2つの標準正規分布乱数を生成
 */
	inline constexpr auto box_muller_transform(const sycl::float2 uniform_floats) noexcept
		-> sycl::float2 {
		float r = sycl::sqrt(-2.0f * sycl::log(uniform_floats.x()));
		// sincos関数でsinとcosを同時に計算
		float scos, ssin;
		ssin = sycl::sincos(2.0f * std::numbers::pi_v<float> * uniform_floats.y(), &scos);
		return {r * scos, r * ssin}; // { z0, z1 }
	}

	/**
 * @brief 並列処理に適したカウンタベースRNG (Philox-2x32-2) の簡易版
 * @param counter 各ワークアイテムに固有の値（グローバルIDなど）
 * @param key シードとして機能する値
 * @return 2つの32ビット乱数
 */
	inline constexpr auto philox_2x32(const u32 counter, const u32 key) noexcept -> sycl::uint2 {
		const u32 M1 = 0xD2511F53;
		const u32 M2 = 0xCD9E8D57;

		// 2ラウンドの置換ネットワーク
		u32 hi = M1 * counter;
		u32 lo = M2 * (counter + key);

		hi = M1 * lo;
		lo = M2 * (hi + key);

		return {lo, hi};
	}
} // namespace sotoba

/// @todo: テスト書く