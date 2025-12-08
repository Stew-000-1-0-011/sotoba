cmake_minimum_required(VERSION 3.28)

project(sotoba
	VERSION 0.0.0
	DESCRIPTION "todo"
	LANGUAGES CXX
)

# --- チェック ---
# if(NOT CMAKE_CXX_COMPILER)
# 	message(FATAL_ERROR "No C++ compiler specified. Please set CMAKE_CXX_COMPILER explicitly (e.g., cmake -DCMAKE_CXX_COMPILER=clang++ ..).")
# endif()

if(NOT (CMAKE_GENERATOR MATCHES "Ninja"))
	message(FATAL_ERROR
		"This project requires the Ninja. "
		"Your current generator is '${CMAKE_GENERATOR}'."
	)
endif()

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


# --- ビルド設定 ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_STATIC_LIB "build static_lib" ON)
option(BUILD_SHARED_LIB "build shared_lib" ON)
option(BUILD_MODULE_LIB "build module_lib" ON)
option(BUILD_DEVS "Build devs" OFF)
set(RELEASE_OPTIMIZE -O2 CACHE STRING "The compiler option for optimization, e.g. -O2 or -Os")


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
if(CMAKE_CXX_COMPILER MATCHES "icpx")
	target_compile_definitions(${PROJECT_NAME}_dep_private
		INTERFACE
			${PROJECT_NAME}_USE_SYCL
			SYCL2020_DISABLE_DEPRECATION_WARNINGS  #どうもコンパイラ内で処理されていそう？ソースコードで定義しても効果なかった
			SYCL_SIMPLE_SWIZZLES
	)
	target_compile_options(${PROJECT_NAME}_dep_private
		INTERFACE
			-fsycl
	)
	target_link_options(${PROJECT_NAME}_dep_private
		INTERFACE
			-fsycl
	)
	target_include_directories(${PROJECT_NAME}_dep_private
		INTERFACE
			/opt/intel/oneapi/2025.3/include/
	)
endif()

add_library(${PROJECT_NAME}_dep_export INTERFACE EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME}_dep_export
	INTERFACE
		# add exported dependencies (s.t. any header depends)
)


# --- 各種ターゲットを作成 ---
set(install_target_list "")

include(cmake/common.cmake)

if(CMAKE_CXX_COMPILER MATCHES "icpx")
	target_compile_options(${PROJECT_NAME}_build_debug_export
		INTERFACE
		-fno-sanitize=all
	)
	target_link_options(${PROJECT_NAME}_build_debug_export
		INTERFACE
		-fno-sanitize=all
	)
endif()

include(cmake/lib.cmake)
# include(cmake/bin.cmake)

if(BUILD_DEVS)
	include(CTest)
	include(FetchContent)

	# --- find package ---
	# find_package(foo REQUIRED)
	# find_package(Pangolin REQUIRED)

	FetchContent_Declare(
		Pangolin
		GIT_REPOSITORY https://github.com/stevenlovegrove/Pangolin.git
		GIT_TAG eb51feaf3f4bddb9768d4f8ac93536407a1779cd
		PATCH_COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/patches/apply_patch.bash" "${CMAKE_CURRENT_SOURCE_DIR}/patches/fix_pangolin.patch"
	)
	FetchContent_MakeAvailable(Pangolin)

	FetchContent_Declare(doctest
		GIT_REPOSITORY https://github.com/doctest/doctest.git
	)
	FetchContent_Populate(doctest)  # doctestのincludeパスが多すぎるので...

	# --- 依存の指定 ---
	add_library(${PROJECT_NAME}_dev_dep INTERFACE EXCLUDE_FROM_ALL)
	target_link_libraries(${PROJECT_NAME}_dev_dep
		INTERFACE
			# add dev dependencies
			pango_display
			pango_plot
			pango_video
	)
	target_include_directories(${PROJECT_NAME}_dev_dep
		INTERFACE
			${doctest_SOURCE_DIR}/doctest/  # doctestのincludeパスが多すぎるので...
	)

	set(install_example_target_list "")
	include(cmake/dev.cmake)
endif()


# --- 個別に追加したいものがあればここで ---
# target_compile_definitions(${PROJECT_NAME}_bin_foo
# 	PRIVATE
# 		FOO
# )


# --- インストール設定 ---
include(GNUInstallDirs)

message("install_target_list: ${install_target_list}")
install(TARGETS
	${install_target_list}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}
)

if(BUILD_DEVS)
	install(TARGETS
		${install_example_target_list}
		RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}/examples/
	)
endif()