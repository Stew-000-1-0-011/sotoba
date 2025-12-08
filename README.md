# sotoba

点群(データ数 N)と、図形(データ数 M)で表される数個のオブジェクトのフィッティングをICPでO(MN)で行うライブラリ。  
図形が少ないなら割と高速。  
数個のオブジェクトのフィッティングができるので、自己位置推定しながらボールの認識をする...とかもできるはず。

SVDによるICPとPoint-to-PlaneのICPを実装。

## 使い方
まだfind_packageできるようにはなってない。
```
# 予め、clang-format-20などが入っているか確認してね

git clone <このリポジトリ>
cd <このリポジトリ>

# git commitやgit push前にフォーマットやビルド/テストのチェックをしてくれるようにする
. setup.bash

# ビルドしたいもの以外をコメントアウトしたりBUILD_DEVSをOFFにしてね
nano build.bash

# icpxに関連するビルド/テストの実行前には`/opt/intel/oneapi/setvars.sh`か以下をソースしてね
. source_setvars.bash

# ビルド
. build.bash

# ビルド、テストの実行
. check_before_push.bash

# ビルドのクリーン
. clean.bash

# examplesの実行(gcc_build, icpx_buildでも同様)
./clang_build/Release/examples/svd_simulation < svd_simulation.txt
./clang_build/Release/examples/normal_known_simulation < normal_known_simulation.txt
```

## 対応環境
Ubuntu24.04
(clang-format-20などとベタ書きしてしまったため。そこらへんを一括置換すればWindowsでも動きそう)

## pre-commit, pre-pushについて
### pre-commit
clang-formatでのフォーマットをする
### pre-push
clang-formatでのフォーマットがされてなければ弾く  
ビルドが通らなければ弾く  
テストが通らなければ弾く  

## clangdについて
コード補完などはclangdを使用。  
ビルド時にcompile_commands.jsonをルートディレクトリに出すようにしているので、ルートディレクトリをclang拡張の入ったVSCodeで開けば補完が効くはず。
