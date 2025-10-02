block(PROPAGATE install_target_list)
	file(GLOB bin_entries LIST_DIRECTORIES true CONFIGURE_DEPENDS src/bin/*)
	file(GLOB bin_main LIST_DIRECTORIES false CONFIGURE_DEPENDS src/main.cpp)
	list(APPEND bin_entries ${bin_main})

	foreach(bin_entry ${bin_entries})
		if(IS_DIRECTORY ${bin_entry})
			file(GLOB bin_sources "${bin_entry}/*.cpp")
			
			get_filename_component(bin_name "${bin_entry}" NAME)
			
			add_executable(${PROJECT_NAME}_bin_${bin_name} ${bin_sources})
			target_link_libraries(${PROJECT_NAME}_bin_${bin_name}
				PRIVATE
					${PROJECT_NAME}_dep_private
					${PROJECT_NAME}_dep_export
					${PROJECT_NAME}_lib
					${PROJECT_NAME}_build_private
					${PROJECT_NAME}_build_export
			)
			set_target_properties(${PROJECT_NAME}_bin_${bin_name} PROPERTIES OUTPUT_NAME ${bin_name})
			list(APPEND install_target_list "${PROJECT_NAME}_bin_${bin_name}")
		elseif(NOT IS_DIRECTORY "${bin_entry}")
			get_filename_component(bin_name "${bin_entry}" NAME_WE)
			message(STATUS "bin_name: ${bin_name}")
			add_executable(${PROJECT_NAME}_bin_${bin_name} ${bin_entry})
			target_link_libraries(${PROJECT_NAME}_bin_${bin_name}
				PRIVATE
					${PROJECT_NAME}_dep_private
					${PROJECT_NAME}_dep_export
					${PROJECT_NAME}_lib
					${PROJECT_NAME}_build_private
					${PROJECT_NAME}_build_export
			)
			set_target_properties("${PROJECT_NAME}_bin_${bin_name}" PROPERTIES OUTPUT_NAME "${bin_name}")
			set_target_properties("${PROJECT_NAME}_bin_${bin_name}" PROPERTIES
				RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debugs/bin"
				RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/releases/bin"
			)
			list(APPEND install_target_list "${PROJECT_NAME}_bin_${bin_name}")
		endif()
	endforeach()
endblock()