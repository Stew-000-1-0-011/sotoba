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

if(NOT (CMAKE_GENERATOR MATCHES "Ninja" OR CMAKE_GENERATOR MATCHES "Makefiles"))
	message(FATAL_ERROR
		"Error: This project requires the Ninja or Makefile generator. "
		"Your current generator is '${CMAKE_GENERATOR}'."
	)
endif()


if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	message(FATAL_ERROR
		"CMAKE_INSTALL_PREFIX must be set explicitly. Default install prefix is not allowed."
		"Please configure with -DCMAKE_INSTALL_PREFIX=<path>"
	)
endif()


# --- ビルド設定 ---
include(CTest)
include(FetchContent)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## RPATH抹消
# RPATH を一切使わないようにする
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "")


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
		# add private dependencies (s.t. no header depends but any source depends)
)
target_compile_definitions(${PROJECT_NAME}_dep_private
	INTERFACE
		# add macro definition if need
		SYCL_SIMPLE_SWIZZLES
)

add_library(${PROJECT_NAME}_dep_export INTERFACE EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME}_dep_export
	INTERFACE
		# add exported dependencies (s.t. any header depends)
		Eigen3::Eigen
)
target_compile_options(${PROJECT_NAME}_dep_export
	INTERFACE
		-fsycl
)
target_link_options(${PROJECT_NAME}_dep_export
	INTERFACE
		-fsycl
)
target_include_directories(${PROJECT_NAME}_dep_export
	INTERFACE
		/opt/intel/oneapi/2025.2/include/
)


# --- 各種ターゲットを作成 ---
set(install_target_list "")

include(cmake/common.cmake)
include(cmake/lib.cmake)
include(cmake/bin.cmake)

if(BUILD_DEVS)

	# --- find package ---
	# find_package(foo REQUIRED)
	find_package(Pangolin REQUIRED)

	FetchContent_Declare(doctest
		GIT_REPOSITORY https://github.com/doctest/doctest.git
	)
	FetchContent_Populate(doctest)  # doctestのincludeパスが多すぎるので...

	# 
	# FetchContent_Declare(pangolin
	# 	GIT_REPOSITORY https://github.com/stevenlovegrove/Pangolin
	# )
	# FetchContent_Populate(pangolin)

	# --- 依存の指定 ---
	add_library(${PROJECT_NAME}_dev_dep INTERFACE EXCLUDE_FROM_ALL)
	target_link_libraries(${PROJECT_NAME}_dev_dep
		INTERFACE
			# add dev dependencies
			pango_display
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

add_executable(${PROJECT_NAME}_expanded_cpp EXCLUDE_FROM_ALL
	expanded_cpp.cpp
)
target_link_libraries(${PROJECT_NAME}_expanded_cpp
	PRIVATE
		${PROJECT_NAME}_dep_private
		${PROJECT_NAME}_dep_export
		${PROJECT_NAME}_build_debug_private
		${PROJECT_NAME}_build_debug_export
)

# --- インストール設定 ---
include(GNUInstallDirs)

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