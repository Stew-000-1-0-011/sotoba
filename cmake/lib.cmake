block(PROPAGATE install_target_list)
	# --- 依存関係の記述 ---
	#find_package(foo REQUIRED)

	file(GLOB lib_sources CONFIGURE_DEPENDS "src/*.cpp")
	list(FILTER lib_sources EXCLUDE REGEX "src/bin/")
	list(FILTER lib_sources EXCLUDE REGEX "src/main.cpp")

	if(lib_sources)
		add_library(${PROJECT_NAME}_lib OBJECT EXCLUDE_FROM_ALL
			${lib_sources}
		)
		target_link_libraries(${PROJECT_NAME}_lib
			PRIVATE
				${PROJECT_NAME}_dep_private
				${PROJECT_NAME}_build_private
			PUBLIC
				${PROJECT_NAME}_dep_export
				${PROJECT_NAME}_build_export
		)
		target_include_directories(${PROJECT_NAME}_lib
			PUBLIC
				$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
				$<INSTALL_INTERFACE:include>
		)

		if(BUILD_STATIC_LIB)
			add_library(${PROJECT_NAME}_static_lib STATIC $<TARGET_OBJECTS:${PROJECT_NAME}_lib>)
			target_link_libraries(${PROJECT_NAME}_static_lib PUBLIC ${PROJECT_NAME}_lib)
			set_target_properties(${PROJECT_NAME}_static_lib PROPERTIES OUTPUT_NAME static_lib)
			set_target_properties(${PROJECT_NAME}_static_lib PROPERTIES
				RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debugs/bin"
				LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debugs/lib"
				ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debugs/lib"
				RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/releases/bin"
				LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/releases/lib"
				ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/releases/lib"
			)
			list(APPEND install_target_list ${PROJECT_NAME}_static_lib)
		endif()

		if(BUILD_SHARED_LIB)
			add_library(${PROJECT_NAME}_shared_lib SHARED $<TARGET_OBJECTS:${PROJECT_NAME}_lib>)
			target_link_libraries(${PROJECT_NAME}_shared_lib PUBLIC ${PROJECT_NAME}_lib)
			set_target_properties(${PROJECT_NAME}_shared_lib PROPERTIES OUTPUT_NAME shared_lib)
			set_target_properties(${PROJECT_NAME}_static_lib PROPERTIES
				RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debugs/bin"
				LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debugs/lib"
				ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debugs/lib"
				RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/releases/bin"
				LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/releases/lib"
				ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/releases/lib"
			)
			list(APPEND install_target_list ${PROJECT_NAME}_shared_lib)
		endif()

		if(BUILD_MODULE_LIB)
			add_library(${PROJECT_NAME}_module_lib MODULE $<TARGET_OBJECTS:${PROJECT_NAME}_lib>)
			target_link_libraries(${PROJECT_NAME}_module_lib PUBLIC ${PROJECT_NAME}_lib)
			set_target_properties(${PROJECT_NAME}_module_lib PROPERTIES OUTPUT_NAME module_lib)
			set_target_properties(${PROJECT_NAME}_static_lib PROPERTIES
				RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debugs/bin"
				LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debugs/lib"
				ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debugs/lib"
				RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/releases/bin"
				LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/releases/lib"
				ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/releases/lib"
			)
			list(APPEND install_target_list ${PROJECT_NAME}_module_lib)
		endif()
	else()
		add_library(${PROJECT_NAME}_lib INTERFACE EXCLUDE_FROM_ALL)
		target_link_libraries(${PROJECT_NAME}_lib
			INTERFACE
				${PROJECT_NAME}_dep_export
				${PROJECT_NAME}_build_export
		)
		target_include_directories(${PROJECT_NAME}_lib
			INTERFACE
				$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
				$<INSTALL_INTERFACE:include>
		)
	endif()
endblock()