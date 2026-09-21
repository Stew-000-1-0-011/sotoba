cmake_minimum_required(VERSION 3.28)

project(sotoba
	VERSION 0.0.0
	DESCRIPTION "Header-only O(MN) point-to-plane ICP library for a few known surfaces"
	LANGUAGES CXX
)

# --- オプション ---
option(BUILD_STATIC_LIB "build static_lib" ON)
option(BUILD_SHARED_LIB "build shared_lib" ON)
option(BUILD_MODULE_LIB "build module_lib" ON)
option(BUILD_DEVS "Build devs" OFF)
option(BUILD_EXAMPLES "Build examples (needs Pangolin, only meaningful when BUILD_DEVS=ON)" ON)
set(RELEASE_OPTIMIZE -O2 CACHE STRING "The compiler option for optimization, e.g. -O2 or -Os")

# --- チェック ---
# if(NOT CMAKE_CXX_COMPILER)
# 	message(FATAL_ERROR "No C++ compiler specified. Please set CMAKE_CXX_COMPILER explicitly (e.g., cmake -DCMAKE_CXX_COMPILER=clang++ ..).")
# endif()

# ヘッダオンリーの利用側 (add_subdirectory / find_package、例えば ament_cmake が
# 既定で使う Unix Makefiles) までジェネレータを縛る理由はないので、Ninja 以外でも
# configure を失敗させない。Ninja はビルドが速く、build.bash 側で複数のコンパイラ
# 構成をまとめて回す都合上おすすめではあるので、トップレベルのときだけ案内する。
if(PROJECT_IS_TOP_LEVEL AND NOT (CMAKE_GENERATOR MATCHES "Ninja"))
	message(STATUS "Generator is '${CMAKE_GENERATOR}'. Ninja is recommended.")
endif()

# add_subdirectory / find_package で使われたときにまで警告を出すと鬱陶しいので、
# トップレベルプロジェクトとして configure されたときだけ出す。
if(PROJECT_IS_TOP_LEVEL)
	if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
		message(WARNING
			"CMAKE_INSTALL_PREFIX should be set explicitly.\n"
			"Please configure with `-DCMAKE_INSTALL_PREFIX=<path>`.\n"
			"set CMAKE_INSTALL_PREFIX=./install"
		)
		unset(CMAKE_INSTALL_PREFIX CACHE)
		set(CMAKE_INSTALL_PREFIX ./install CACHE STRING "CMAKE_INSTALL_PREFIX")
	endif()

	if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
		message(WARNING
			"CMAKE_BUILD_TYPE should be set explicitly.\n"
			"Please configure with `-DCMAKE_BUILD_TYPE=<Debug or Release>`.\n"
			"set CMAKE_BUILD_TYPE=Debug"
		)
		unset(CMAKE_BUILD_TYPE CACHE)
		set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build.")
	endif()
endif()


# --- ビルド設定 ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# CMAKE_INSTALL_INCLUDEDIR 等を sotoba_headers ターゲット定義より前に使うため、
# ここで先に include しておく (後段の install(...) からも同じものを使う)。
include(GNUInstallDirs)

# --- find_package(dev用のものはdev.cmake内で) ---
# find_package(foo REQUIRED)
find_package(Eigen3 REQUIRED)

# --- 依存を指定 ---
## libとbinの依存が共通 -> binはあくまでlibに従うもの、libの使わないものを使いたいならプロジェクトを分けるべき
add_library(${PROJECT_NAME}_dep_private INTERFACE EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME}_dep_private
	INTERFACE
		Eigen3::Eigen
)

add_library(${PROJECT_NAME}_dep_export INTERFACE EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME}_dep_export
	INTERFACE
		# add exported dependencies (s.t. any header depends)
)


# --- 各種ターゲットを作成 ---
set(install_target_list "")

include(cmake/common.cmake)

include(cmake/lib.cmake)
# include(cmake/bin.cmake)


# --- find_package(sotoba) 用のクリーンな INTERFACE ターゲット ---
# 注意: サニタイザ (sotoba_build_export) や -march=native (sotoba_build_common_private) 等の
# 開発用フラグを絶対に載せないこと。また既存の *_lib / *_dep_* / *_build_* 系ターゲットは
# 全て EXCLUDE_FROM_ALL で export set に入っていないため、リンクすると
# install(EXPORT) が失敗する。よってここでは include ディレクトリ・cxx_std_23・Eigen3::Eigen
# だけを持つ独立したターゲットを新規に作る。
add_library(${PROJECT_NAME}_headers INTERFACE)
target_include_directories(${PROJECT_NAME}_headers
	INTERFACE
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(${PROJECT_NAME}_headers INTERFACE cxx_std_23)
target_link_libraries(${PROJECT_NAME}_headers INTERFACE Eigen3::Eigen)
# install(EXPORT ... NAMESPACE sotoba::) はターゲット名をそのまま名前空間化するため、
# EXPORT_NAME を指定しないと find_package 経由では sotoba::sotoba_headers になってしまい、
# 下の add_subdirectory 用エイリアス (sotoba::sotoba) と名前が食い違う。
# 利用側が find_package と add_subdirectory のどちらでも同じ名前を書けるように揃える。
set_target_properties(${PROJECT_NAME}_headers PROPERTIES EXPORT_NAME ${PROJECT_NAME})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME}_headers)


if(BUILD_DEVS)
	include(CTest)
	include(FetchContent)

	# --- find package ---
	# find_package(foo REQUIRED)
	# find_package(Pangolin REQUIRED)

	if(BUILD_EXAMPLES)
		FetchContent_Declare(
			Pangolin
			GIT_REPOSITORY https://github.com/stevenlovegrove/Pangolin.git
			GIT_TAG eb51feaf3f4bddb9768d4f8ac93536407a1779cd
			PATCH_COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/patches/apply_patch.bash" "${CMAKE_CURRENT_SOURCE_DIR}/patches/fix_pangolin.patch"
		)
		FetchContent_MakeAvailable(Pangolin)
	endif()

	FetchContent_Declare(doctest
		GIT_REPOSITORY https://github.com/doctest/doctest.git
	)
	FetchContent_Populate(doctest)  # doctestのincludeパスが多すぎるので...

	# --- 依存の指定 ---
	# tests は doctest のみに依存させる (Pangolin 無しでテストだけ回せるように)
	add_library(${PROJECT_NAME}_test_dep INTERFACE EXCLUDE_FROM_ALL)
	target_include_directories(${PROJECT_NAME}_test_dep
		INTERFACE
			${doctest_SOURCE_DIR}/doctest/  # doctestのincludeパスが多すぎるので...
	)

	# examples は Pangolin (重い) + doctest に依存させる
	if(BUILD_EXAMPLES)
		add_library(${PROJECT_NAME}_example_dep INTERFACE EXCLUDE_FROM_ALL)
		target_link_libraries(${PROJECT_NAME}_example_dep
			INTERFACE
				pango_display
				pango_plot
				pango_video
		)
		target_include_directories(${PROJECT_NAME}_example_dep
			INTERFACE
				${doctest_SOURCE_DIR}/doctest/  # doctestのincludeパスが多すぎるので...
		)
	endif()

	set(install_example_target_list "")
	include(cmake/dev.cmake)
endif()


# --- 個別に追加したいものがあればここで ---
# target_compile_definitions(${PROJECT_NAME}_bin_foo
# 	PRIVATE
# 		FOO
# )


# --- インストール設定 ---
# (GNUInstallDirs は sotoba_headers ターゲット定義のために既に include 済み)

message("install_target_list: ${install_target_list}")
if(install_target_list)
	install(TARGETS
		${install_target_list}
		ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
		RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}
	)
endif()

if(BUILD_DEVS AND BUILD_EXAMPLES AND install_example_target_list)
	install(TARGETS
		${install_example_target_list}
		RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}/examples/
	)
endif()

# --- find_package(sotoba) のためのパッケージ化 ---
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS ${PROJECT_NAME}_headers EXPORT ${PROJECT_NAME}Targets)

install(EXPORT ${PROJECT_NAME}Targets
	FILE ${PROJECT_NAME}Targets.cmake
	NAMESPACE ${PROJECT_NAME}::
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
	${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
	INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
write_basic_package_version_file(
	${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMajorVersion
)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
	${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
