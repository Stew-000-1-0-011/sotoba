#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/src/Cholesky/LLT.h>
#include <Eigen/src/Core/util/Constants.h>

#include "sotoba/math/sym_mat.hpp"
#include "sotoba/math/vec_forward_decl.hpp"
#include "sotoba/stdtypes.hpp"
#include "sotoba/surf_obj_id.hpp"

#include "sotoba/math/quaternion.hpp"
#include "sotoba/math/se3.hpp"
#include "sotoba/math/square_mat.hpp"
#include "sotoba/math/vec.hpp"
#include "sotoba/surface/surface.hpp"

#include "resource.hpp"

namespace sotoba::icp_resource::normal_known_icp_impl {
	using math::SE3;
	using math::SquareMat;
	using math::SymMat;
	using math::UVec3;
	using math::Vec;
	using math::Vec3;
	using math::Vec4;
	using Vec6 = Vec<6>;
	namespace vec = math::vec;
	using surface::ExplanationOnlySurface;
	using surface::surfacelike;
	using icp_resource::IcpError;
	using icp_resource::ObjStatus;

	/// LiDAR の点ごとの誤差モデル。
	/// 点対面残差の分散を σ_r² cos² + r² σ_θ² (1 - cos²) で見積もる。
	struct NoiseModel final {
		/// ビーム方向(距離方向)のノイズ標準偏差 [m]。
		float sigma_range;
		/// 角度ノイズ標準偏差 [rad]。横方向の位置誤差は r * sigma_angle になる。
		float sigma_angle;
	};

	/// run_icp の重み付け設定。既定 (両方とも無指定) では全点の重みが 1 になり、
	/// 重み付けを入れる前と完全に同一の挙動になる。
	struct IcpWeighting final {
		/// 無指定なら全点の重みを 1 とする (ノイズモデルによる重み付けを行わない)。
		std::optional<NoiseModel> noise{};
		/// Huber カーネルの閾値 k (正規化残差に対する)。無指定ならロバスト化しない。
		/// noise が無指定の場合、正規化残差は生の残差 [m] そのものになるので、
		/// k の単位も [m] になる点に注意。
		std::optional<float> huber_k{};
	};

	template <surfacelike... Surfaces_>
	struct NormalKnownResource final {
		// 表面とその情報、座標変換後の表面のバッファ
		std::tuple<std::vector<Surfaces_>...> surfs;
		std::tuple<std::vector<Surfaces_>...> moved_surfs;
		std::array<std::vector<ObjSurfId>, sizeof...(Surfaces_)> osids;

		// 点群とその最近接点に関する情報
		// Vec4: [x,y,z,距離]
		std::vector<std::pair<std::pair<Vec4, UVec3>, ObjSurfId>> qs;

		// 加算されていくやつら
		std::vector<Vec6> b;
		std::vector<SymMat<3>> a_w;
		std::vector<SymMat<3>> a_t;
		std::vector<SquareMat<3>> a_wt;
		std::vector<usize> counts;
		// 直近の run_icp における、オブジェクトごとの重みの総和 Σ w_i
		std::vector<float> weight_sums;

		// ここに入れた姿勢をもとに、ICPがはしり、補正された結果がここに入る
		std::vector<SE3> obj_poses;

		// 直近の run_icp におけるオブジェクトごとの姿勢更新結果
		std::vector<ObjStatus> obj_statuses;
		// 直近の run_icp で実際に回ったループ回数
		u32 loop_count;

		u8 obj_num;

		NormalKnownResource(
			std::tuple<std::vector<Surfaces_>...>&& surfs,
			std::array<std::vector<ObjSurfId>, sizeof...(Surfaces_)>&& osids,
			const u8 obj_num,
			const usize points_num
		) noexcept
			: surfs{std::move(surfs)}
			, moved_surfs{}
			, osids{std::move(osids)}
			, qs{}
			, b{}
			, a_w{}
			, a_t{}
			, a_wt{}
			, counts{}
			, weight_sums{}
			, obj_poses{}
			, obj_statuses{}
			, loop_count{0}
			, obj_num{obj_num} {
			(
				[&]<surfacelike S_>() {
					std::get<std::vector<S_>>(this->moved_surfs) =
						std::get<std::vector<S_>>(this->surfs);
				}.template operator()<Surfaces_>(),
				...
			);
			this->qs.resize(points_num);
			this->b.resize(obj_num);
			this->a_w.resize(obj_num);
			this->a_t.resize(obj_num);
			this->a_wt.resize(obj_num);
			this->counts.resize(obj_num);
			this->weight_sums.resize(obj_num, 0.f);
			this->obj_poses.resize(obj_num, SE3::ide());
			this->obj_statuses.resize(obj_num, ObjStatus::not_run);
		}

		decltype(auto) obj_pose(this auto&& self, const u8 oid) noexcept {
			return self.obj_poses[oid];
		}

		// このバッファが受け入れられる最大点数
		auto points_capacity() const noexcept -> usize {
			return this->qs.size();
		}

		// 直近の run_icp におけるオブジェクトの姿勢更新結果
		auto obj_status(const u8 oid) const noexcept -> ObjStatus {
			return this->obj_statuses[oid];
		}

		// 直近の run_icp におけるオブジェクトの対応点数
		auto correspondence_count(const u8 oid) const noexcept -> usize {
			return this->counts[oid];
		}

		/// 直近の run_icp における、このオブジェクトの重みの総和 Σ w_i。
		/// 重み付けが無効なときは対応点数と一致する。
		/// information_matrix() を正規化したい場合の分母になる。
		auto weight_sum(const u8 oid) const noexcept -> float {
			return this->weight_sums[oid];
		}

		// 直近の run_icp で実際に回ったループ回数 (常に max_loop_num 以下)
		auto last_loop_count() const noexcept -> u32 {
			return this->loop_count;
		}

		/// 直近の run_icp における、このオブジェクトの正規方程式の係数行列
		/// A = Σ JᵀNJ (情報行列)。添字 0..2 が回転 w、3..5 が並進 t。
		///
		/// 対応点数での正規化も tikhonov 正則化も加えていない **生の総和**。
		/// Σ/N が欲しければ correspondence_count(oid) で割ること
		/// (weighting.noise を与えた場合は weight_sum(oid) で割ること)。
		///
		/// weighting.noise を与えなかった場合、各点の寄与は無重み (w=1) の
		/// ままなので従来どおり素の総和であり、共分散は Cov ≒ σ² A⁻¹ として
		/// σ² (センサの距離ノイズ分散) を呼び出し側が与える必要がある。
		/// weighting.noise を与えた場合、各点の寄与にはすでに 1/σ_i² が
		/// 重みとして掛かっているため、A はそのまま **真の情報行列** になり、
		/// Cov ≒ A⁻¹ がそのまま使える (呼び出し側が別途 σ² を与える必要はない)。
		/// tikhonov を含めないのは、正則化が入ると
		/// 縮退方向で不確かさを過小評価してしまうため。
		///
		/// 値は最後に回ったイテレーションのもの。last_loop_count() == 0 のとき
		/// (max_loop_num == 0 を渡した場合、および IcpError::too_many_points で
		/// 抜けた場合) は直前の run_icp の値が残っているので参照しないこと。
		auto information_matrix(const u8 oid) const noexcept -> SymMat<6> {
			SymMat<6> ret{};
			for (u8 i = 0; i < 3; ++i)
				for (u8 j = i; j < 3; ++j) {
					ret[i, j] = this->a_w[oid][i, j];
					ret[i + 3, j + 3] = this->a_t[oid][i, j];
				}
			for (u8 i = 0; i < 3; ++i)
				for (u8 j = 0; j < 3; ++j) { ret[i, j + 3] = this->a_wt[oid][i, j]; }
			return ret;
		}

		/// 同じく正規方程式の右辺 b = Σ JᵀNe。こちらも生の総和。
		auto residual_vector(const u8 oid) const noexcept -> Vec6 {
			return this->b[oid];
		}

		/// 姿勢を更新するのに必要な最小の対応点数。
		///
		/// 点対面ICPでは対応点1つが1本のスカラー拘束になるので、SE3 の6自由度を
		/// 決めるには最低6点が要る。
		///
		/// ただしこれは必要条件にすぎず十分条件ではない。例えば1枚の平面に
		/// 正対した点は何点集めても法線方向の並進1自由度しか拘束しないため、
		/// 対応点数が足りていても係数行列がランク落ちすることはある
		/// (tikhonov を入れるとコレスキー分解は必ず成功してしまうので、
		///  縮退はこの閾値では検出できない)。
		/// 姿勢が実際にどれだけ拘束されているかを見たい場合は
		/// information_matrix() の固有値を調べること。
		static constexpr usize min_correspondences = 6;

		/// 点対面ICPを走らせ、obj_poses を更新する。
		///
		/// point_cloud はセンサ座標系の点群。面の可視性判定はセンサ原点(0,0,0)を
		/// 基準に行うため、obj_pose には「マップ座標系の形状をセンサ座標系へ写す変換」
		/// (= 自己位置の逆変換) を入れること。向きを取り違えると全点が不可視になる。
		///
		/// accept_distance2 は対応点として受け入れる距離の **二乗**。
		///
		/// point_cloud.size() が points_capacity() を超える場合、バッファの再確保は
		/// 行わず IcpError::too_many_points を返す。このとき姿勢・状態は一切変化しない。
		///
		/// ループ回数は max_loop_num をハード上限とし、これを超えて回ることはない。
		/// 姿勢を更新した全オブジェクトの更新量 delta2 の最大値が convergence_delta2
		/// 以下になった時点で打ち切るため、実際の回数は常に max_loop_num 以下になる
		/// (last_loop_count() で取得できる)。
		/// 更新されたオブジェクトが1つも無い場合 (全て too_few_correspondences や
		/// solve_failed の場合) は最大値が 0 のままなので、既定の
		/// convergence_delta2 = 0.f でも1回で打ち切られる。姿勢が動かない以上
		/// 回し続けても結果は変わらないため、これは意図した挙動。
		///
		/// 内部で組む正規方程式の係数行列と右辺は information_matrix() /
		/// residual_vector() で取得できる。どちらも対応点数での正規化や
		/// tikhonov 正則化を加える前の生の総和であり、ObjStatus によらず
		/// 最後に回ったイテレーションの値になる。
		///
		/// weighting は点ごとの重み付けの設定。既定の `IcpWeighting{}`
		/// (noise, huber_k とも無指定) では全点の重みが厳密に 1 になり、
		/// 重み付けを導入する前と完全に同一の挙動・数値結果になる。
		/// weighting.noise を与えると、点対面残差の分散を
		/// σ_r²cos² + r²σ_θ²(1-cos²) で見積もり、その逆数を重みとして使う
		/// (センサ距離に比例して大きくなる横方向誤差を考慮したノイズモデル)。
		/// weighting.huber_k を与えると、正規化残差 e/σ に対して Huber の
		/// IRLS 重みをさらに掛け、外れ値(動物体・誤対応など)の影響を抑える。
		/// weighting が不正な値 (負の σ、0 以下の huber_k、非有限値) の場合は
		/// IcpError::invalid_weighting を返す。このとき呼び出しは何も行わず、
		/// 姿勢も状態も変化しない。
		///
		/// accept_distance2_begin は対応距離ゲートを反復内で粗→細に絞る
		/// (coarse-to-fine) ためのパラメータ。
		/// - `accept_distance2_begin <= 0.f` (既定値) のときはスケジュールしない。
		///   全反復で accept_distance2 を使う、従来と完全に同一の挙動になる。
		/// - `accept_distance2_begin > 0.f` のときは、1回目の反復で
		///   accept_distance2_begin、最終反復で accept_distance2 になるよう
		///   **等比数列**でゲートを絞る。最終反復のゲートは浮動小数点の
		///   累積誤差を避けるため accept_distance2 に厳密に一致させる
		///   (等比の積算値ではなく、呼び出し側が指定した値そのものを使う)。
		///   初期姿勢の誤差が (絞る前の) 対応距離ゲートを超えると誤対応に
		///   固着しやすいため、広いゲートから始めることで収束半径を広げられる。
		///
		/// **反復回数の上限 max_loop_num は増えない**。coarse-to-fine は
		/// 既存の反復予算の中でゲートを絞るだけであり、追加の反復は行わない。
		///
		/// accept_distance2_begin が非有限、または
		/// `0.f < accept_distance2_begin < accept_distance2` (狭い→広いの
		/// 逆順、呼び出し側のバグの可能性が高い) の場合は
		/// IcpError::invalid_accept_schedule を返す。このとき呼び出しは
		/// 何も行わず、姿勢も状態も変化しない。
		///
		/// **早期打ち切りとの相互作用**: convergence_delta2 による早期打ち切りは、
		/// スケジュールが有効な間はゲートがまだ粗い段階で発動しうるため、
		/// 現在のゲートが accept_distance2 に達している反復 (スケジュール無効時は
		/// 常に、スケジュール有効時は最終反復のみ) でのみ許可する。
		/// そのため、スケジュールを有効にすると早期打ち切りは実質無効になる
		/// (最終反復でしか判定されず、break しても回る回数は変わらない)。
		auto run_icp(
			std::span<const Vec3> point_cloud,
			const Vec6& tikhonov,
			const u32 max_loop_num,
			const float accept_distance2,
			const float convergence_delta2 = 0.f,
			const IcpWeighting& weighting = {},
			const float accept_distance2_begin = 0.f
		) noexcept -> IcpError {
			if (point_cloud.size() > this->qs.size()) return IcpError::too_many_points;

			if (weighting.noise) {
				const auto& noise = *weighting.noise;
				if (!(noise.sigma_range >= 0.f) || !math::isfinite(noise.sigma_range))
					return IcpError::invalid_weighting;
				if (!(noise.sigma_angle >= 0.f) || !math::isfinite(noise.sigma_angle))
					return IcpError::invalid_weighting;
			}
			if (weighting.huber_k) {
				const float k = *weighting.huber_k;
				if (!(k > 0.f) || !math::isfinite(k)) return IcpError::invalid_weighting;
			}
			if (!math::isfinite(accept_distance2_begin))
				return IcpError::invalid_accept_schedule;
			if (accept_distance2_begin > 0.f && accept_distance2_begin < accept_distance2)
				return IcpError::invalid_accept_schedule;

			const auto tikhonov_w = vec::split<0, 3>(tikhonov);
			const auto tikhonov_t = vec::split<3, 6>(tikhonov);

			this->loop_count = 0;

			// 対応距離ゲートの coarse-to-fine スケジュール (ループの外で一度だけ計算)。
			// accept_distance2_begin <= 0.f なら scheduled=false のままで、
			// current_gate2 は常に accept_distance2 そのものになる
			// (従来・重み付け導入前と完全に同一の数値結果を保つため)。
			const bool scheduled = (accept_distance2_begin > 0.f) && (max_loop_num > 1);
			const float gate_ratio = scheduled
				? std::exp(
					  std::log(accept_distance2 / accept_distance2_begin)
					  / static_cast<float>(max_loop_num - 1)
				  )
				: 1.f;
			float gate2 = scheduled ? accept_distance2_begin : accept_distance2;

			for (u32 iloop = 0; iloop < max_loop_num; ++iloop) {
				this->loop_count = iloop + 1;

				// 最終反復では浮動小数点の累積誤差を避け、呼び出し側が指定した
				// accept_distance2 を厳密に使う。
				const float current_gate2 =
					(iloop + 1 == max_loop_num) ? accept_distance2 : gate2;

				// surfsをobj_posesに従い移動
				[&]<usize... idxs_>(std::index_sequence<idxs_...>) {
					(
						[&]<surfacelike S_>(
							const std::vector<S_>& surf,
							const std::vector<ObjSurfId>& osid,
							std::vector<S_>& moved_surf
						) {
							for (usize i = 0; i < surf.size(); ++i) {
								const auto [oid, sid] = osid_depack(osid[i]);
								moved_surf[i] = surf[i];
								moved_surf[i].apply_se3(this->obj_poses[u8(oid)]);
							}
						}(std::get<idxs_>(this->surfs),
						  this->osids[idxs_],
						  std::get<idxs_>(this->moved_surfs)),
						...
					);
				}(std::index_sequence_for<Surfaces_...>{});

				// 各点の最近接点をqsに格納
				[&]<usize... idxs_>(std::index_sequence<idxs_...>) {
					for (usize ip = 0; ip < point_cloud.size(); ++ip) {
						if (!vec::isfinite(point_cloud[ip])) {
							this->qs[ip] = {
								{{Vec3{}, Vec{std::numeric_limits<float>::infinity()}}, {}},
								ObjSurfId::Null
							};
							continue;
						}

						std::pair<std::pair<Vec4, UVec3>, ObjSurfId> q{
							{{Vec3{}, Vec{std::numeric_limits<float>::infinity()}}, {}},
							ObjSurfId::Null
						};
						(
							[&]<surfacelike S_>(const std::vector<S_>& surf, const u8 isurf_kind) {
								for (usize isurf = 0; isurf < surf.size(); ++isurf) {
									const auto q_ = surf[isurf].closest_pdn(point_cloud[ip]);
									if (q_.first.w() < q.first.first.w()) {
										q = {q_, this->osids[isurf_kind][isurf]};
									}
								}
							}(std::get<idxs_>(this->moved_surfs), idxs_),
							...
						);

						this->qs[ip] = q;
					}
				}(std::index_sequence_for<Surfaces_...>{});

				// Ax = bのA, bを計算
				for (u8 iobj = 0; iobj < this->obj_num; ++iobj) {
					this->b[iobj] = Vec6{};
					this->a_w[iobj] = SymMat<3>{};
					this->a_t[iobj] = SymMat<3>{};
					this->a_wt[iobj] = SquareMat<3>{};
					this->counts[iobj] = 0;
					this->weight_sums[iobj] = 0.f;
				}
				for (usize ip = 0; ip < point_cloud.size(); ++ip) {
					const auto [qdn, osid] = this->qs[ip];
					if (osid == ObjSurfId::Null) continue;
					const auto [qd, n] = qdn;
					const auto q = qd.xyz();
					const auto d = qd.w();
					if (current_gate2 < d) {
						this->qs[ip].second = ObjSurfId::Null;
						continue;
					}
					const Vec3 p = point_cloud[ip];
					const float err_n = vec::dot((p - q), n);
					const Vec3 p_c = vec::cross(p, n);

					// --- 点ごとの重み ---
					float w = 1.f;
					float sigma2 = 1.f; // 正規化残差を作るための分散 (noise 無指定なら 1)
					if (weighting.noise) {
						const float r2 = vec::dot(p, p);
						const float np = vec::dot(n, p); // |n| = 1 なので cos² = np²/r²
						const float cos2 = (r2 > float(math::epsilon)) ? (np * np / r2) : 1.f;
						const float sr2 = math::pow2(weighting.noise->sigma_range);
						const float st2 = math::pow2(weighting.noise->sigma_angle);
						// 0除算を避けるための純粋な数値ガード (モデル上の意味は無い)
						sigma2 =
							std::max(sr2 * cos2 + r2 * st2 * (1.f - cos2), float(math::epsilon));
						w = 1.f / sigma2;
					}
					if (weighting.huber_k) {
						const float k = *weighting.huber_k;
						const float s2 = math::pow2(err_n) / sigma2; // 正規化残差の二乗
						if (math::pow2(k) < s2) { w *= k / math::sqrt(s2); }
					}

					const u8 iobj = std::to_underlying(osid_depack(osid).first);
					this->b[iobj] += w * Vec6{err_n * p_c, err_n * n};
					this->a_w[iobj] += w * vec::self_dyad(p_c);
					this->a_t[iobj] += w * vec::self_dyad(n);
					this->a_wt[iobj] += w * vec::dyad(p_c, n);
					this->counts[iobj]++;
					this->weight_sums[iobj] += w;
				}

				// 次の反復に向けてゲートを等比で絞る。current_gate2 の計算は
				// 常に accept_distance2 / gate2 の三項演算で行うため、この乗算の
				// 丸め誤差が current_gate2 に影響することはない。
				gate2 *= gate_ratio;

				float max_delta2 = 0.f;
				for (u8 iobj = 0; iobj < this->obj_num; ++iobj) {
					if (this->counts[iobj] < min_correspondences) {
						// 点が少なすぎるオブジェクトはスキップ
						this->obj_statuses[iobj] = ObjStatus::too_few_correspondences;
						continue;
					}
					// 正規化と tikhonov 正則化は Eigen 側を組むときにだけ適用する。
					// a_w / a_t / a_wt / b は生の総和 (Σ) のまま残し、
					// information_matrix() / residual_vector() から素の情報行列を
					// 取れるようにする。
					const float n = this->weight_sums[iobj];

					// コレスキー分解、w, tを求める
					using Matrix6f = Eigen::Matrix<float, 6, 6>;

					Matrix6f a_tri;
					for (u8 i = 0; i < 3; ++i)
						for (u8 j = i; j < 3; ++j) {
							a_tri(i, j) =
								this->a_w[iobj][i, j] / n + (i == j ? tikhonov_w[i] : 0.f);
							a_tri(i + 3, j + 3) =
								this->a_t[iobj][i, j] / n + (i == j ? tikhonov_t[i] : 0.f);
						}
					for (u8 i = 0; i < 3; ++i)
						for (u8 j = 0; j < 3; ++j) {
							a_tri(i, j + 3) = this->a_wt[iobj][i, j] / n;
						}
					const Matrix6f a = a_tri.selfadjointView<Eigen::Upper>();
					Eigen::LLT<Matrix6f> cholesky(a);
					if (cholesky.info() != Eigen::Success) {
						this->obj_statuses[iobj] = ObjStatus::solve_failed;
						continue;
					}

					Eigen::Vector<float, 6> b_;
					for (u8 i = 0; i < 6; ++i) b_(i) = this->b[iobj][i] / n;

					const auto x = cholesky.solve(b_);
					const SE3 diff =
						SE3{math::UQuaternion{vec::fast_normalize(Vec4{x[0], x[1], x[2], 2.f})},
							{x[3], x[4], x[5]}};

					// 推定姿勢を更新
					this->obj_poses[iobj] = (diff * this->obj_poses[iobj]).normalize();
					this->obj_statuses[iobj] = ObjStatus::updated;

					// 早期打ち切り判定用のdelta2 (w, tそれぞれのdotの和)
					const Vec3 w{x[0], x[1], x[2]};
					const Vec3 t{x[3], x[4], x[5]};
					const float delta2 = vec::dot(w, w) + vec::dot(t, t);
					if (max_delta2 < delta2) max_delta2 = delta2;
				}

				// 早期打ち切り: 姿勢を更新した全オブジェクトのdelta2の最大値が
				// convergence_delta2以下ならループを抜ける。max_loop_numがハード上限。
				// ただし、ゲートがまだ粗い段階(スケジュール有効時の最終反復以外)で
				// 発動すると粗い解のまま終わってしまうため、現在のゲートが
				// accept_distance2 に到達している反復でのみ判定する。
				if (!scheduled || iloop + 1 == max_loop_num) {
					if (max_delta2 <= convergence_delta2) break;
				}
			}

			return IcpError::none;
		}
	};

	static_assert(icp_resource::icp_resource<
				  NormalKnownResource<ExplanationOnlySurface<0>, ExplanationOnlySurface<1>>,
				  ExplanationOnlySurface<0>,
				  ExplanationOnlySurface<1>>);
	// 面の種類が2つ以外でも成立すること (icp_resource concept が種類数を
	// ハードコードしていないことの担保)。
	static_assert(icp_resource::icp_resource<
				  NormalKnownResource<ExplanationOnlySurface<0>>,
				  ExplanationOnlySurface<0>>);
	static_assert(icp_resource::icp_resource<
				  NormalKnownResource<
					  ExplanationOnlySurface<0>,
					  ExplanationOnlySurface<1>,
					  ExplanationOnlySurface<2>>,
				  ExplanationOnlySurface<0>,
				  ExplanationOnlySurface<1>,
				  ExplanationOnlySurface<2>>);
} // namespace sotoba::icp_resource::normal_known_icp_impl

namespace sotoba::icp_resource {
	using normal_known_icp_impl::IcpWeighting;
	using normal_known_icp_impl::NoiseModel;
	using normal_known_icp_impl::NormalKnownResource;
}

#ifdef sotoba_ENABLE_TESTING
	#include <array>
	#include <cmath>
	#include <limits>
	#include <vector>

	#include <doctest.h>

	#include "sotoba/math/approx_check.hpp"
	// テストが surface::Rectangle を使うので、include 順に依存せず
	// 自己完結するようここで include しておく。
	#include "sotoba/surface/rectangle.hpp"

TEST_SUITE("normal_known_icp.hpp") {
	using namespace sotoba;
	using math::SE3;
	using math::UVec3;
	using math::Vec;
	using math::Vec3;
	using math::Vec4;
	using math::Vec6;
	namespace vec = math::vec;
	using surface::Rectangle;
	using icp_resource::IcpError;
	using icp_resource::IcpWeighting;
	using icp_resource::NoiseModel;
	using icp_resource::NormalKnownResource;
	using icp_resource::ObjStatus;
	using math::ApproxCheck;

	// 原点(0,0,0)から見えるよう、ローカル座標系の中心を(0,0,0)に置いた矩形。
	// obj_poseで(0,0,5)へ移動させるとrectangle.hppのテストと同じ配置になる。
	inline auto forward_rect() -> Rectangle {
		return Rectangle{
			Vec3{0.f, 0.f, 0.f},
			Vec4{1.f, 0.f, 0.f, 2.f},
			Vec4{0.f, 1.f, 0.f, 1.f},
			UVec3{0.f, 0.f, -1.f},
		};
	}

	// 法線が原点と逆向きなので、obj_poseをどう動かしても可視化されない矩形。
	inline auto backward_rect() -> Rectangle {
		return Rectangle{
			Vec3{0.f, 0.f, 0.f},
			Vec4{1.f, 0.f, 0.f, 2.f},
			Vec4{0.f, 1.f, 0.f, 1.f},
			UVec3{0.f, 0.f, 1.f},
		};
	}

	// obj_num=1, 面1枚のNormalKnownResourceを作る
	inline auto make_icp(const Rectangle& rect, const usize capacity)
		-> NormalKnownResource<Rectangle> {
		std::array<std::vector<ObjSurfId>, 1> osids{
			std::vector<ObjSurfId>{osid_pack(ObjId(0), SurfId(0))}
		};
		return NormalKnownResource<Rectangle>{
			std::tuple{std::vector<Rectangle>{rect}},
			std::move(osids),
			1,
			capacity
		};
	}

	// 矩形ローカル座標系の面上の点を、true_poseでセンサ座標系へ写した
	// (ノイズ無しの)点群を作る。u:[-1.4,1.4], v:[-0.7,0.7] の範囲は
	// forward_rect() の半辺長(2.0, 1.0)に収まるのでクランプされない。
	inline auto sample_points(const SE3& true_pose) -> std::vector<Vec3> {
		std::vector<Vec3> pts;
		for (int iu = -2; iu <= 2; ++iu) {
			for (int iv = -2; iv <= 2; ++iv) {
				const Vec3 local{float(iu) * 0.7f, float(iv) * 0.35f, 0.f};
				pts.push_back(true_pose.app_v(local));
			}
		}
		return pts;
	}

	TEST_CASE("run_icp: 点数がちょうど容量ならIcpError::noneが返る") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f}));
		REQUIRE(points.size() == 25);

		const auto err = icp.run_icp(std::span{points}, Vec6{}, 1, 100.f);

		CHECK(err == IcpError::none);
	}

	TEST_CASE("run_icp: 点数が容量を1つでも超えるとtoo_many_pointsが返り状態が不変") {
		auto icp = make_icp(forward_rect(), 24);
		const auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f}));
		REQUIRE(points.size() == 25);

		const SE3 seed = SE3::trans(Vec3{0.1f, 0.f, 4.5f});
		icp.obj_pose(0) = seed;

		const auto err = icp.run_icp(std::span{points}, Vec6{}, 5, 100.f);

		CHECK(err == IcpError::too_many_points);
		CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		CHECK(icp.last_loop_count() == 0);
		CHECK(icp.obj_status(0) == ObjStatus::not_run);
	}

	TEST_CASE("run_icp: max_loop_num=0なら姿勢が変化せず回った回数は0") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f}));

		const SE3 seed = SE3::trans(Vec3{0.1f, 0.f, 4.5f});
		icp.obj_pose(0) = seed;

		const auto err = icp.run_icp(std::span{points}, Vec6{}, 0, 100.f);

		CHECK(err == IcpError::none);
		CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		CHECK(icp.last_loop_count() == 0);
	}

	TEST_CASE("run_icp: 回った回数は常にmax_loop_num以下") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f}));
		icp.obj_pose(0) = SE3::trans(Vec3{0.f, 0.f, 4.5f});

		const Vec6 tikhonov{0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f};
		constexpr u32 max_loop_num = 50;
		const auto err = icp.run_icp(std::span{points}, tikhonov, max_loop_num, 100.f);

		CHECK(err == IcpError::none);
		CHECK(icp.last_loop_count() <= max_loop_num);
	}

	TEST_CASE("run_icp: convergence_delta2を大きく与えるとmax_loop_numより少ない回数で打ち切られる") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f}));
		icp.obj_pose(0) = SE3::trans(Vec3{0.f, 0.f, 4.5f});

		const Vec6 tikhonov{0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f};
		constexpr u32 max_loop_num = 50;
		const auto err =
			icp.run_icp(std::span{points}, tikhonov, max_loop_num, 100.f, 1e6f);

		CHECK(err == IcpError::none);
		CHECK(icp.last_loop_count() < max_loop_num);
	}

	TEST_CASE("run_icp: 既知形状に対しずらしたシードが正解姿勢へ近づく") {
		auto icp = make_icp(forward_rect(), 25);
		const SE3 true_pose = SE3::trans(Vec3{0.f, 0.f, 5.f});
		const auto points = sample_points(true_pose);

		const SE3 seed = SE3::trans(Vec3{0.f, 0.f, 4.5f});
		icp.obj_pose(0) = seed;

		const Vec6 tikhonov{0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f};
		const auto err = icp.run_icp(std::span{points}, tikhonov, 50, 100.f);

		CHECK(err == IcpError::none);
		CHECK(icp.obj_status(0) == ObjStatus::updated);

		const float seed_err = std::fabs(seed.p.z() - true_pose.p.z());
		const float result_err = std::fabs(icp.obj_pose(0).p.z() - true_pose.p.z());
		CHECK(result_err < seed_err);
		CHECK(result_err < 0.05f);
	}

	TEST_CASE("run_icp: どの面にも対応しない点群はtoo_few_correspondencesになり姿勢はシードのまま") {
		auto icp = make_icp(backward_rect(), 4);
		const std::vector<Vec3> points{
			Vec3{0.f, 0.f, 1.f},
			Vec3{0.f, 0.f, 2.f},
			Vec3{0.f, 0.f, 3.f},
			Vec3{0.f, 0.f, 4.f},
		};

		const SE3 seed = SE3::trans(Vec3{1.f, 2.f, 3.f});
		icp.obj_pose(0) = seed;

		const auto err = icp.run_icp(std::span{points}, Vec6{}, 5, 100.f);

		CHECK(err == IcpError::none);
		CHECK(icp.obj_status(0) == ObjStatus::too_few_correspondences);
		CHECK(icp.correspondence_count(0) == 0);
		CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
	}

	TEST_CASE("run_icp: 非有限な点が混ざってもクラッシュせず対応点として採用されない") {
		auto icp = make_icp(forward_rect(), 27);
		auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f})); // 25点
		points.push_back(Vec3{std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f});
		points.push_back(Vec3{std::numeric_limits<float>::infinity(), 0.f, 0.f});
		REQUIRE(points.size() == 27);

		icp.obj_pose(0) = SE3::trans(Vec3{0.f, 0.f, 4.5f});

		const Vec6 tikhonov{0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f};
		const auto err = icp.run_icp(std::span{points}, tikhonov, 3, 100.f);

		CHECK(err == IcpError::none);
		CHECK(icp.qs[25].second == ObjSurfId::Null);
		CHECK(icp.qs[26].second == ObjSurfId::Null);
		CHECK(icp.correspondence_count(0) <= 25);
	}

	TEST_CASE("information_matrix: tikhonovを変えても値が変わらない(正則化が焼き込まれていないこと)") {
		const auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f}));
		const SE3 seed = SE3::trans(Vec3{0.05f, 0.f, 4.5f});

		auto icp_zero = make_icp(forward_rect(), 25);
		icp_zero.obj_pose(0) = seed;
		const auto err_zero = icp_zero.run_icp(std::span{points}, Vec6{}, 1, 100.f);

		auto icp_big = make_icp(forward_rect(), 25);
		icp_big.obj_pose(0) = seed;
		const Vec6 big_tikhonov{1e6f, 1e6f, 1e6f, 1e6f, 1e6f, 1e6f};
		const auto err_big = icp_big.run_icp(std::span{points}, big_tikhonov, 1, 100.f);

		REQUIRE(err_zero == IcpError::none);
		REQUIRE(err_big == IcpError::none);
		// a_w/a_tが実際に積み上がった状態であること (tikhonov=0では正対した
		// 矩形の面内並進・法線周り回転が不可観測でsolve_failedになるが、
		// 生の総和自体はコレスキーの成否によらず積み上がっている)。
		REQUIRE(icp_zero.correspondence_count(0) >= 3);
		REQUIRE(icp_big.correspondence_count(0) >= 3);

		const auto im_zero = icp_zero.information_matrix(0);
		const auto im_big = icp_big.information_matrix(0);
		for (u8 i = 0; i < 6; ++i)
			for (u8 j = i; j < 6; ++j) { CHECK(im_zero[i, j] == im_big[i, j]); }
	}

	TEST_CASE("run_icp: 対応点が min_correspondences 未満なら姿勢を更新しない") {
		// 点対面の対応点1つ = スカラー拘束1本なので、SE3 の6自由度を決めるには
		// 最低 6 点が要る。境界 (5点 / 6点) をまたいで挙動が変わることを確認する。
		static_assert(NormalKnownResource<Rectangle>::min_correspondences == 6);

		const SE3 true_pose = SE3::trans(Vec3{0.f, 0.f, 5.f});
		const auto points = sample_points(true_pose); // 25点
		// 1枚の平面はランク落ちするので、コレスキーを成功させるため tikhonov を入れる
		const Vec6 tikhonov{0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f};

		SUBCASE("5点では更新されず、姿勢は呼び出し時の値のまま") {
			auto icp = make_icp(forward_rect(), 25);
			const SE3 seed = SE3::trans(Vec3{0.f, 0.f, 4.9f});
			icp.obj_pose(0) = seed;

			const auto err =
				icp.run_icp(std::span{points}.subspan(0, 5), tikhonov, 3, 100.f);

			CHECK(err == IcpError::none);
			CHECK(icp.correspondence_count(0) == 5);
			CHECK(icp.obj_status(0) == ObjStatus::too_few_correspondences);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}

		SUBCASE("6点なら更新される") {
			auto icp = make_icp(forward_rect(), 25);
			const SE3 seed = SE3::trans(Vec3{0.f, 0.f, 4.9f});
			icp.obj_pose(0) = seed;

			const auto err =
				icp.run_icp(std::span{points}.subspan(0, 6), tikhonov, 3, 100.f);

			CHECK(err == IcpError::none);
			CHECK(icp.correspondence_count(0) == 6);
			CHECK(icp.obj_status(0) == ObjStatus::updated);
			CHECK_FALSE(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}
	}

	TEST_CASE("information_matrix: 対称性") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f}));
		icp.obj_pose(0) = SE3::trans(Vec3{0.05f, 0.f, 4.5f});

		const Vec6 tikhonov{0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f};
		const auto err = icp.run_icp(std::span{points}, tikhonov, 1, 100.f);
		REQUIRE(err == IcpError::none);

		const auto im = icp.information_matrix(0);
		for (u8 i = 0; i < 6; ++i)
			for (u8 j = 0; j < 6; ++j) { CHECK(im[i, j] == im[j, i]); }
	}

	TEST_CASE("information_matrix: 正対した点群では生の総和がa_tブロックの手計算値と一致する") {
		auto icp = make_icp(forward_rect(), 25);
		const SE3 true_pose = SE3::trans(Vec3{0.f, 0.f, 5.f});
		const auto points = sample_points(true_pose);
		icp.obj_pose(0) = true_pose; // シードなしで完全一致させる

		const auto err = icp.run_icp(std::span{points}, Vec6{}, 1, 100.f);
		REQUIRE(err == IcpError::none);
		// tikhonov=0だと面内並進・法線周り回転が不可観測でsolve_failedになるが、
		// a_w/a_t/a_wtの生の総和はコレスキーの成否によらず積み上がっている。
		REQUIRE(icp.correspondence_count(0) == 25);

		const auto im = icp.information_matrix(0);
		// forward_rect()に正対しているので法線は全点(0,0,-1)。
		// a_t = Σ self_dyad(n) = N * diag(0,0,1) (生の総和、正規化前)。
		CHECK(im[3, 3] == 0.f);
		CHECK(im[4, 4] == 0.f);
		CHECK(im[5, 5] == 25.f);
		CHECK(im[3, 4] == 0.f);
		CHECK(im[3, 5] == 0.f);
		CHECK(im[4, 5] == 0.f);

		// 対応点数で割っても0にならないこと(正規化版として妥当)。
		const float n = float(icp.correspondence_count(0));
		CHECK(im[5, 5] / n > 0.f);
	}

	TEST_CASE("information_matrix: too_few_correspondencesでも生の総和(0除算やゴミではない)が読める") {
		auto icp = make_icp(backward_rect(), 4);
		const std::vector<Vec3> points{
			Vec3{0.f, 0.f, 1.f},
			Vec3{0.f, 0.f, 2.f},
			Vec3{0.f, 0.f, 3.f},
			Vec3{0.f, 0.f, 4.f},
		};
		icp.obj_pose(0) = SE3::trans(Vec3{1.f, 2.f, 3.f});

		const Vec6 tikhonov{10.f, 10.f, 10.f, 10.f, 10.f, 10.f};
		const auto err = icp.run_icp(std::span{points}, tikhonov, 5, 100.f);

		REQUIRE(err == IcpError::none);
		REQUIRE(icp.obj_status(0) == ObjStatus::too_few_correspondences);
		REQUIRE(icp.correspondence_count(0) == 0);

		// 対応点が0なので生の総和もクラッシュせず0のまま(tikhonovが焼き込まれていない)。
		const auto im = icp.information_matrix(0);
		for (u8 i = 0; i < 6; ++i)
			for (u8 j = i; j < 6; ++j) { CHECK(im[i, j] == 0.f); }

		const auto res = icp.residual_vector(0);
		for (u8 i = 0; i < 6; ++i) { CHECK(res[i] == 0.f); }
	}

	TEST_CASE("residual_vector: 読み取れる") {
		auto icp = make_icp(forward_rect(), 25);
		const SE3 true_pose = SE3::trans(Vec3{0.f, 0.f, 5.f});
		const auto points = sample_points(true_pose);
		icp.obj_pose(0) = true_pose; // 完全一致 -> 各点の誤差は0

		const auto err = icp.run_icp(std::span{points}, Vec6{}, 1, 100.f);
		REQUIRE(err == IcpError::none);
		// b の生の総和はコレスキーの成否によらず積み上がっているので、
		// 対応点数で確認する(tikhonov=0だとこの配置はsolve_failedになりうる)。
		REQUIRE(icp.correspondence_count(0) == 25);

		const auto res = icp.residual_vector(0);
		for (u8 i = 0; i < 6; ++i) { CHECK(res[i] == doctest::Approx(0.f).epsilon(1e-4)); }
	}

	// --- IcpWeighting ---

	TEST_CASE("run_icp: 既定のIcpWeighting{}ではweight_sumが対応点数と厳密に一致する") {
		auto icp = make_icp(forward_rect(), 25);
		const SE3 true_pose = SE3::trans(Vec3{0.f, 0.f, 5.f});
		const auto points = sample_points(true_pose);
		icp.obj_pose(0) = SE3::trans(Vec3{0.05f, 0.f, 4.5f});

		const Vec6 tikhonov{0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f};
		const auto err = icp.run_icp(std::span{points}, tikhonov, 1, 100.f);

		REQUIRE(err == IcpError::none);
		REQUIRE(icp.correspondence_count(0) == 25);
		// 重み付け無効時は weight_sum が counts の厳密な float 表現になる
		// (w が厳密に1.0fのまま積み上がるため)。
		CHECK(icp.weight_sum(0) == float(icp.correspondence_count(0)));
	}

	TEST_CASE("run_icp: ノイズモデルが入射角で効く(正対 vs 斜め)") {
		// 正対/傾いたそれぞれの配置で sigma_angle を 0 -> 大 にしたときの
		// information_matrix (trace) の変化率を比べる。
		// 正対(cos²≈1)ではほぼ変わらず、斜め(cos²が小さい点を含む)では大きく下がる方向。
		auto trace_of = [](const Rectangle& rect, const SE3& true_pose, const float sigma_angle) {
			auto icp = make_icp(rect, 25);
			const auto points = sample_points(true_pose);
			icp.obj_pose(0) = true_pose; // シードなしで完全一致 (残差0でも重みは効く)

			const IcpWeighting weighting{
				.noise = NoiseModel{.sigma_range = 0.05f, .sigma_angle = sigma_angle}
			};
			const auto err = icp.run_icp(std::span{points}, Vec6{}, 1, 100.f, 0.f, weighting);
			REQUIRE(err == IcpError::none);
			REQUIRE(icp.correspondence_count(0) >= 3);

			const auto im = icp.information_matrix(0);
			float trace = 0.f;
			for (u8 i = 0; i < 6; ++i) trace += im[i, i];
			return trace;
		};

		// 正対: forward_rect に真正面から (センサ~面の距離5m)
		const SE3 straight_pose = SE3::trans(Vec3{0.f, 0.f, 5.f});
		// 斜め: y軸まわりに約0.9radピッチさせ、法線の多くがビーム方向からずれるようにする
		const SE3 tilted_pose =
			SE3{math::quaternion::ypr(Vec3{0.f, 0.9f, 0.f}), Vec3{0.f, 0.f, 5.f}};

		const float trace_straight_0 = trace_of(forward_rect(), straight_pose, 0.f);
		const float trace_straight_1 = trace_of(forward_rect(), straight_pose, 0.5f);
		const float trace_tilted_0 = trace_of(forward_rect(), tilted_pose, 0.f);
		const float trace_tilted_1 = trace_of(forward_rect(), tilted_pose, 0.5f);

		REQUIRE(trace_straight_0 > 0.f);
		REQUIRE(trace_tilted_0 > 0.f);

		const float ratio_straight = trace_straight_1 / trace_straight_0;
		const float ratio_tilted = trace_tilted_1 / trace_tilted_0;

		// 正対時は sigma_angle を増やしてもあまり落ちず、斜め時は大きく落ちる。
		CHECK(ratio_tilted < ratio_straight);
	}

	TEST_CASE("run_icp: ノイズモデルで遠い点の重みが落ちる(grazing)") {
		// 大きく傾けた(grazingな)配置で、sigma_angleを0から正にすると
		// information_matrixが小さくなること (1/r^2減衰のような回転消去は起きない)。
		auto icp0 = make_icp(forward_rect(), 25);
		auto icp1 = make_icp(forward_rect(), 25);

		const SE3 grazing_pose =
			SE3{math::quaternion::ypr(Vec3{0.f, 1.0f, 0.f}), Vec3{0.f, 0.f, 5.f}};
		const auto points = sample_points(grazing_pose);
		icp0.obj_pose(0) = grazing_pose;
		icp1.obj_pose(0) = grazing_pose;

		const IcpWeighting weighting0{.noise = NoiseModel{.sigma_range = 0.05f, .sigma_angle = 0.f}
		};
		const IcpWeighting weighting1{
			.noise = NoiseModel{.sigma_range = 0.05f, .sigma_angle = 0.3f}
		};

		const auto err0 = icp0.run_icp(std::span{points}, Vec6{}, 1, 100.f, 0.f, weighting0);
		const auto err1 = icp1.run_icp(std::span{points}, Vec6{}, 1, 100.f, 0.f, weighting1);
		REQUIRE(err0 == IcpError::none);
		REQUIRE(err1 == IcpError::none);
		REQUIRE(icp0.correspondence_count(0) >= 3);
		REQUIRE(icp1.correspondence_count(0) >= 3);

		const auto im0 = icp0.information_matrix(0);
		const auto im1 = icp1.information_matrix(0);
		float trace0 = 0.f, trace1 = 0.f;
		for (u8 i = 0; i < 6; ++i) {
			trace0 += im0[i, i];
			trace1 += im1[i, i];
		}
		CHECK(trace1 < trace0);
		// weight_sum自体も落ちていること
		CHECK(icp1.weight_sum(0) < icp0.weight_sum(0));
	}

	TEST_CASE("run_icp: 【本命】Huberが外れ値に効く") {
		// 正しい点群に、面から大きく飛び出た外れ値点を数点混ぜる。
		// 同じ点群・同じシード・同じループ回数で、huber_k無指定/指定を比較する。
		const SE3 true_pose = SE3::trans(Vec3{0.f, 0.f, 5.f});
		auto points = sample_points(true_pose); // 25点、正しい点群
		// 面から大きく(2m)飛び出した外れ値を3点混ぜる (u,v範囲内なので対応点にはなる)。
		points.push_back(true_pose.app_v(Vec3{0.5f, 0.3f, 2.0f}));
		points.push_back(true_pose.app_v(Vec3{-0.5f, -0.3f, 2.0f}));
		points.push_back(true_pose.app_v(Vec3{0.0f, 0.6f, 2.0f}));
		REQUIRE(points.size() == 28);

		const SE3 seed = SE3::trans(Vec3{0.f, 0.f, 4.5f});
		const Vec6 tikhonov{0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f};
		constexpr u32 max_loop_num = 30;

		auto icp_no_huber = make_icp(forward_rect(), 28);
		icp_no_huber.obj_pose(0) = seed;
		const auto err_no_huber =
			icp_no_huber.run_icp(std::span{points}, tikhonov, max_loop_num, 100.f);

		auto icp_huber = make_icp(forward_rect(), 28);
		icp_huber.obj_pose(0) = seed;
		const IcpWeighting weighting{.huber_k = 0.1f};
		const auto err_huber = icp_huber.run_icp(
			std::span{points},
			tikhonov,
			max_loop_num,
			100.f,
			0.f,
			weighting
		);

		REQUIRE(err_no_huber == IcpError::none);
		REQUIRE(err_huber == IcpError::none);

		const float err_z_no_huber = std::fabs(icp_no_huber.obj_pose(0).p.z() - true_pose.p.z());
		const float err_z_huber = std::fabs(icp_huber.obj_pose(0).p.z() - true_pose.p.z());

		// huber無指定 -> 外れ値に引っ張られ、真値から大きくずれる
		CHECK(err_z_no_huber > 0.1f);
		// huber指定 -> ずれが明確に小さくなる
		CHECK(err_z_huber < 0.05f);
		// 本命: Huberありの方が誤差が小さい
		CHECK(err_z_huber < err_z_no_huber);
	}

	TEST_CASE("run_icp: weightingが不正ならinvalid_weightingが返り姿勢が不変") {
		auto icp = make_icp(forward_rect(), 25);
		const auto points = sample_points(SE3::trans(Vec3{0.f, 0.f, 5.f}));
		const SE3 seed = SE3::trans(Vec3{0.f, 0.f, 4.5f});

		SUBCASE("sigma_rangeが負") {
			icp.obj_pose(0) = seed;
			const IcpWeighting weighting{
				.noise = NoiseModel{.sigma_range = -0.1f, .sigma_angle = 0.f}
			};
			const auto err = icp.run_icp(std::span{points}, Vec6{}, 5, 100.f, 0.f, weighting);
			CHECK(err == IcpError::invalid_weighting);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
			CHECK(icp.obj_status(0) == ObjStatus::not_run);
		}

		SUBCASE("sigma_angleが負") {
			icp.obj_pose(0) = seed;
			const IcpWeighting weighting{
				.noise = NoiseModel{.sigma_range = 0.f, .sigma_angle = -0.1f}
			};
			const auto err = icp.run_icp(std::span{points}, Vec6{}, 5, 100.f, 0.f, weighting);
			CHECK(err == IcpError::invalid_weighting);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}

		SUBCASE("huber_kが0") {
			icp.obj_pose(0) = seed;
			const IcpWeighting weighting{.huber_k = 0.f};
			const auto err = icp.run_icp(std::span{points}, Vec6{}, 5, 100.f, 0.f, weighting);
			CHECK(err == IcpError::invalid_weighting);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}

		SUBCASE("huber_kが負") {
			icp.obj_pose(0) = seed;
			const IcpWeighting weighting{.huber_k = -1.f};
			const auto err = icp.run_icp(std::span{points}, Vec6{}, 5, 100.f, 0.f, weighting);
			CHECK(err == IcpError::invalid_weighting);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}

		SUBCASE("sigma_rangeがNaN") {
			icp.obj_pose(0) = seed;
			const IcpWeighting weighting{
				.noise =
					NoiseModel{
						.sigma_range = std::numeric_limits<float>::quiet_NaN(), .sigma_angle = 0.f
					}
			};
			const auto err = icp.run_icp(std::span{points}, Vec6{}, 5, 100.f, 0.f, weighting);
			CHECK(err == IcpError::invalid_weighting);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}

		SUBCASE("huber_kがNaN") {
			icp.obj_pose(0) = seed;
			const IcpWeighting weighting{.huber_k = std::numeric_limits<float>::quiet_NaN()};
			const auto err = icp.run_icp(std::span{points}, Vec6{}, 5, 100.f, 0.f, weighting);
			CHECK(err == IcpError::invalid_weighting);
			CHECK(ApproxCheck{icp.obj_pose(0)} == ApproxCheck{seed});
		}
	}

	TEST_CASE("run_icp: sigma_range=0,sigma_angle=0でもNaN/infにならない(epsilonガード)") {
		auto icp = make_icp(forward_rect(), 25);
		const SE3 true_pose = SE3::trans(Vec3{0.f, 0.f, 5.f});
		const auto points = sample_points(true_pose);
		icp.obj_pose(0) = SE3::trans(Vec3{0.05f, 0.f, 4.5f});

		const Vec6 tikhonov{0.001f, 0.001f, 0.001f, 0.001f, 0.001f, 0.001f};
		const IcpWeighting weighting{.noise = NoiseModel{.sigma_range = 0.f, .sigma_angle = 0.f}};
		const auto err = icp.run_icp(std::span{points}, tikhonov, 3, 100.f, 0.f, weighting);

		CHECK(err == IcpError::none);
		CHECK(vec::isfinite(icp.obj_pose(0).p));
		CHECK(std::isfinite(icp.weight_sum(0)));
		const auto im = icp.information_matrix(0);
		for (u8 i = 0; i < 6; ++i)
			for (u8 j = i; j < 6; ++j) { CHECK(std::isfinite(im[i, j])); }
	}
}

#endif