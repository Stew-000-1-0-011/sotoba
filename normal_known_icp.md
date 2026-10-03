`Cost = sum |n \cdot (Rq + t - p)|^2 = sum (Rq + t - p)^t N (Rq + t - p)`
`R ~ I + S(w)`  (`S(・)`は左から外積を加える行列)
`Cost ~ sum 二次形式{(I + S(w))q + t - p, N} = sum 二次形式{S(-q)w + t - (p - q), N}`
`x := 縦並び{t, w}, J := 横並び{I, S(-q)}, e := p - q`  (並進が先、回転が後。1.2 参照)
`Cost ~ sum 二次形式{Jx - e, N}`
微分して、
`dCdx ~ sum J^tN(Jx - e)`これが0になるときCostは最小。よって
`(sum J^tNJ)x = (sum J^tNe)`
Jをばらす(sumを略す)
`p_c := S(p) \cross n, e_n = e \cdot n`
`J^tNJ = 縦並び{横並び{self_dyad(n), dyad(n, p_c)}, 横並び{dyad(n, p_c)^t, self_dyad(p_c)}}`
`J^tNe = 縦並び{e_n n, e_n p_c}`

## 事前分布つきの定式化

### 1.1 摂動の規約
姿勢`T`はmap->sensor変換。摂動は左(センサ系)。
`T = exp(ξ) T_bar, ξ ∈ se(3)`
事前分布は`ξ ~ N(0, Σ)`、情報行列`Λ = Σ^{-1}`。`ObjPrior::information`は`mean = T_bar`まわりのこの座標での`Λ`。

### 1.2 接空間の成分順序
Sophusの規約に合わせて`Sophus::SE3f::Tangent = (upsilon, omega)`、**並進が先、回転が後**。
`information_matrix()`、`residual_vector()`、`posterior_information()`、`tikhonov`、`ObjPrior::information`は全てこの順。
添字0..2が並進、3..5が回転。
ヤコビアンも`J_i = 横並び{n^t, (p × n)^t}`の順。

### 1.3 正規方程式
`A = sum w_i J_i^t J_i`, `b = sum w_i e_i J_i^t`は重み付きの生の総和で、点数や重みの総和では割らない。
`w_i = 1/σ_i^2`(`NoiseModel`)とすると`A`はFisher情報行列になり、`Λ`(1/m^2, 1/rad^2)と同じ単位で足せる。
このため、非ゼロの事前を使うときは`weighting.noise`を必須にしている。

反復`k`の推定`T_k`に対し
`r = log(T_k T_bar^{-1})`(左事前残差)
`Jinv = SE3f::leftJacobianInverse(r)`
`H = A + Jinv^t Λ Jinv + diag(tikhonov)`
`g = b - Jinv^t Λ r`
`x = H^{-1} g`(Cholesky)
`T_{k+1} = exp(x) T_k`

導出: 更新後の事前残差は`log(exp(x) exp(r)) ~ r + Jinv x`。
事前項のコスト`½|r + Jinv x|^2_Λ`の勾配は`Jinv^t Λ (r + Jinv x)`、データ項の勾配は`A x - b`。
足して0とおくと`(A + Jinv^t Λ Jinv) x = b - Jinv^t Λ r`となる。
**`g`の事前項はマイナス**。
`tikhonov`はLM減衰であり、右辺には何も足さない(事前分布とは別概念)。

`Jinv`が`I`で近似できるのは`r`が小さいときだけで、誤差は`|r|`の1次で効く。
`leftJacobianInverse`は回転角`θ = 2π`に極を持つので、`r`の回転成分が`π`を超える(または非有限の)とき`prior_residual_too_large`を返す。

`Λ = 0`のオブジェクトは事前なしと同じ式を通り、結果は一致する。

### 1.4 物体座標系で与えられた事前
呼び出し側が物体自身の座標系の摂動`T = T_bar exp(ξ_b)`で`Λ_b`を持つとき、`ξ = Ad(T_bar) ξ_b`なので
`Λ = Ad(T_bar^{-1})^t Λ_b Ad(T_bar^{-1})`
`prior_information_from_body(mean, Λ_b)`がこれを計算する。

### 事前による対応点数の緩和
点対面の拘束はスカラー1本なので、事前なしでは6点未満を足切りしている(`min_correspondences`)。
非ゼロの事前を持つオブジェクトは`H`が正定値になるので1点から解く(`min_correspondences_with_prior`)。
1面しか見えず`A`がランク落ちする場合も、見える方向は観測、見えない方向は事前で決まる。

### 事後情報行列
`posterior_information(oid)`は最終反復の`H`。最終反復の線形化点まわりの左摂動座標なので、収束していれば`obj_pose(oid)`まわりの`Λ`としてそのままフレーム間で使える。
対応付けの誤りや地図誤差を含まない楽観的な値である。
