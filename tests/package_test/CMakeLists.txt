# find_package(sotoba) が実際に動作することを確認するための、
# sotoba を外部パッケージとして使う最小の利用側 (consumer) プロジェクト。
#
# sotoba本体の CMakeLists.txt からは add_subdirectory されない、独立した
# プロジェクトとして構成している (ament_cmake など、実際の利用側が
# find_package(sotoba) する状況を模している)。CIや手元での検証用に使う。
#
# 使い方:
#   cmake -S tests/package_test -B <build-dir> -GNinja \
#       -DCMAKE_PREFIX_PATH=<sotobaのインストール先>
#   cmake --build <build-dir>
cmake_minimum_required(VERSION 3.28)

project(sotoba_package_test LANGUAGES CXX)

find_package(sotoba REQUIRED)

# 注意: install(EXPORT ... NAMESPACE sotoba::) はターゲット名 sotoba_headers を
# そのまま名前空間化するため、find_package(sotoba) 経由では
# sotoba::sotoba_headers という名前になる (sotoba::sotoba ではない)。
# sotoba::sotoba は add_subdirectory(sotoba) で使う場合のローカルエイリアス。
add_executable(sotoba_package_test main.cpp)
target_link_libraries(sotoba_package_test PRIVATE sotoba::sotoba_headers)
target_compile_features(sotoba_package_test PRIVATE cxx_std_23)
