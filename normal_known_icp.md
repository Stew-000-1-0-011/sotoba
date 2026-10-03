`Cost = sum |n \cdot (Rq + t - p)|^2 = sum (Rq + t - p)^t N (Rq + t - p)`
`IcpWeighting`を使う場合は各項に`w_i`(スカラー)が掛かる。以降はNを`w_i N`と読み替えればそのまま成立する。
`R ~ I + S(w)`  (`S(・)`は左から外積を加える行列)
`Cost ~ sum 二次形式{(I + S(w))q + t - p, N} = sum 二次形式{S(-q)w + t - (p - q), N}`
`x := 縦並び{w, t}, J := 横並び{S(-q), I}, e := p - q`
`Cost ~ sum 二次形式{Jx - e, N}`
微分して、
`dCdx ~ sum J^tN(Jx - e)`これが0になるときCostは最小。よって
`(sum J^tNJ)x = (sum J^tNe)`
Jをばらす(sumを略す)
`p_c := S(p) \cross n, e_n = e \cdot n`
`J^tNJ = 縦並び{横並び{self_dyad(p_c), dyad(p_c, n)}, 横並び{dyad(p_c, n)^t, self_dyad(n)}}`
`J^tNe = 縦並び{e_n p_c, e_n n}`
