#pragma once

#include <chrono>
#include <thread>

namespace sotoba::sim {
	struct LapTimer final {
		using Clock = std::chrono::system_clock;

		std::chrono::time_point<Clock> last{Clock::now()};

		auto lap() noexcept -> std::chrono::duration<float> {
			const auto now = Clock::now();
			const auto ret = now - this->last;
			this->last = now;
			return ret;
		}

		void clear() noexcept {
			this->last = Clock::now();
		}

		template<class Rep_, class Period>
		void sleep_for(const std::chrono::duration<Rep_, Period>& duration) noexcept {
			const auto now = Clock::now();
			const auto elapsed = now - this->last;
			if(elapsed < duration) {
				std::this_thread::sleep_for(duration - elapsed);
			}
			this->clear();
		}
	};
}