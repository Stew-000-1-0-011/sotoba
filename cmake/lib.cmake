block(PROPAGATE install_target_list)
	# --- 依存関係の記述 ---
	#find_package(foo REQUIRED)

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
endblock()
