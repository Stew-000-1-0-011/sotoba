# sotoba

点群(データ数 N)と、図形(データ数 M)で表される数個のオブジェクトのフィッティングをICPでO(MN)で行うライブラリ。  
図形が少ないなら割と高速。  
数個のオブジェクトのフィッティングができるので、自己位置推定しながらボールの認識をする...とかもできるはず。

Point-to-PlaneのICPを実装。

## 使い方
`find_package(sotoba)` で外部プロジェクトから使える (実例は tests/package_test を参照)。
このリポジトリ自体をビルド・テストする場合は以下の手順で。
```
# 予め、clang-format-20などが入っているか確認してね

git clone <このリポジトリ>
cd <このリポジトリ>

# git commitやgit push前にフォーマットやビルド/テストのチェックをしてくれるようにする
. setup.bash

# ビルドしたいもの以外をコメントアウトしたりBUILD_DEVSをOFFにしてね
nano build.bash

# ビルド
. build.bash

# ビルド、テストの実行
. check_before_push.bash

# ビルドのクリーン
. clean.bash

# examplesの実行(gcc_buildでも同様)
./clang_build/Release/examples/normal_known_simulation < normal_known_simulation.txt
```

## 対応環境
Ubuntu24.04
(clang-format-20などとベタ書きしてしまったため。そこらへんを一括置換すればWindowsでも動きそう)

### 必要なコンパイラ
C++23のうち deducing this (P0847) と多次元`operator[]` (P2128)、`<format>`を使う。
動作確認済みの最低バージョンは **GCC 14** / **Clang 18**。
Ubuntu 24.04の既定の`g++`はGCC 13でdeducing thisが使えないので、
`-DCMAKE_CXX_COMPILER=g++-14`のように明示すること。
満たさないコンパイラでは`sotoba/stdtypes.hpp`が`#error`で弾く。

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

## ディレクトリ構成
Cargoを微妙にまねている。  
- .githooks/  
  pre-commit, pre-pushのためのbashスクリプト
- cmake/  
  CMakeLists.txtの中で呼ばれるcmakeスクリプト
- **examples/**  
  サンプルコード。sotobaを使う側の人はまずこれを読んでみてほしい
- **include/**  
  ヘッダ。sotobaの処理はだいたいヘッダに書いてある
  - math/  
    数学の諸関数/型
  - random/  
    シミュレーション用の乱数
  - sim/  
    シミュレーション用の諸コード
  - **surface/**  
    オブジェクトを構成する曲面が入っている。  
    新たな曲面を使いたい場合、ここにコードを追加してね
  - **icp_resource/**  
    ICPをループ内で呼ぶ場合、毎回メモリ確保などをするのは望ましくない。  
    そこで、ICPに必要な資源を纏めた型を用意した。この値を生成してから、各ループではrun_icpを呼んでほしい  
    (**詳しくはexamplesを見てね！**)
- tests  
  テストコードが入っている。doctestを使っている
  - package_test/  
    `find_package(sotoba)` が実際に動作することを確認するための、
    sotoba を外部パッケージとして使う最小の利用側(consumer)プロジェクト。
    sotoba本体の CMakeLists.txt からは add_subdirectory されない。CIや
    手元での検証で `cmake -S tests/package_test -B <build> -DCMAKE_PREFIX_PATH=<installdir>`
    のように単独で configure して使う。
- .clang*  
  clangツール用の諸設定ファイル。うまく使ってほしい
- memo.md  
  私が書いたメモ。todoなどがある(GitHub Issueにしろはそう)
- *.txt  
  examples下の実行に必要な標準入力のプリセット。意味はexamplesコードを読んで確認してね
- *.bash  
  上の「使い方」参照
