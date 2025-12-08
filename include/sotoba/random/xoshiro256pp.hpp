#pragma once

#include <cstdlib>
#include <limits>
#include "sotoba/stdtypes.hpp"

namespace sotoba::random::xoshiro256pp_impl {
	// 参考: https://prng.di.unimi.it/
	struct Xoshiro256pp {
		u64 s[4];

		// 回転処理
		static inline u64 rotl(const u64 x, u32 k) {
			return (x << k) | (x >> (64 - k));
		}

		// コンストラクタ（シード指定）
		Xoshiro256pp(u64 seed = 0) {
			// SplitMix64で初期シードを拡散して状態を作る
			u64 z = (seed + 0x9e3779b97f4a7c15ULL);
			z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
			z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
			s[0] = z ^ (z >> 31);

			z = (s[0] + 0x9e3779b97f4a7c15ULL);
			z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
			z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
			s[1] = z ^ (z >> 31);

			s[2] = s[1] * 3; // 簡易初期化
			s[3] = s[0] * 7;
		}

		// 次の乱数を取得 (最大値: 2^64-1)
		inline u64 next() {
			const u64 result = rotl(s[0] + s[3], 23) + s[0];
			const u64 t = s[1] << 17;

			s[2] ^= s[0];
			s[0] ^= s[1];
			s[1] ^= s[2];
			s[0] ^= s[3];
			s[2] ^= t;
			s[3] = rotl(s[3], 45);

			return result;
		}

		// [0.0, 1.0) の浮動小数を生成（標準的な変換）
		inline float next_float() {
			return (next() >> 11) * 0x1.0p-53f;
		}

		// 【重要】ジャンプ関数
		// 乱数列を 2^128 ステップ一気に進める。
		// これにより、異なるスレッドで重複しない系列を保証する。
		void jump() {
			static const u64 JUMP[] =
				{0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c};
			u64 s0 = 0;
			u64 s1 = 0;
			u64 s2 = 0;
			u64 s3 = 0;
			for (u32 i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
				for (u32 b = 0; b < 64; b++) {
					if (JUMP[i] & (1ULL << b)) {
						s0 ^= s[0];
						s1 ^= s[1];
						s2 ^= s[2];
						s3 ^= s[3];
					}
					next();
				}
			s[0] = s0;
			s[1] = s1;
			s[2] = s2;
			s[3] = s3;
		}

		static constexpr auto min() noexcept -> u64 {
			return 0;
		}

		static constexpr auto max() noexcept -> u64 {
			return std::numeric_limits<u64>::max();
		}

		constexpr auto operator()() noexcept -> u64 {
			return this->next();
		}
	};
} // namespace sotoba::random::xoshiro256pp_impl

namespace sotoba::random {
	using xoshiro256pp_impl::Xoshiro256pp;
}