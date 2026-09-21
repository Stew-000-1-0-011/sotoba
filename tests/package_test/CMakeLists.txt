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

# sotoba 側で EXPORT_NAME を揃えてあるので、find_package(sotoba) でも
# add_subdirectory(sotoba) でも同じ sotoba::sotoba を書けばよい。
add_executable(sotoba_package_test main.cpp)
target_link_libraries(sotoba_package_test PRIVATE sotoba::sotoba)
target_compile_features(sotoba_package_test PRIVATE cxx_std_23)
