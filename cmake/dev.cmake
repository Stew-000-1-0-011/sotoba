block(PROPAGATE install_example_target_list)
	# --- テスト ---
	enable_testing()

	file(GLOB test_sources CONFIGURE_DEPENDS "tests/*.cpp")
	if(test_sources)
		add_executable(${PROJECT_NAME}_tests EXCLUDE_FROM_ALL
			${test_sources}
		)
		target_link_libraries(${PROJECT_NAME}_tests
			PRIVATE
				${PROJECT_NAME}_dep_private
				${PROJECT_NAME}_dep_export
				${PROJECT_NAME}_dev_dep
				${PROJECT_NAME}_lib
				${PROJECT_NAME}_build_private
				${PROJECT_NAME}_build_export
		)
		target_compile_definitions(${PROJECT_NAME}_tests
			PRIVATE
				${PROJECT_NAME}_ENABLE_TESTING
		)
		set_target_properties(${PROJECT_NAME}_tests PROPERTIES OUTPUT_NAME tests)
		set_target_properties(${PROJECT_NAME}_tests PROPERTIES
			RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debugs/tests/"
			RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/releases/tests/"
		)

		message("add test:")
		add_test(NAME
			tests COMMAND $<TARGET_FILE:${PROJECT_NAME}_tests>
		)

		# testsはinstallしない
	endif()


	# --- examplesのビルド ---
	file(GLOB exam_entries CONFIGURE_DEPENDS "examples/*")
	foreach(exam_entry ${exam_entries})
		if(IS_DIRECTORY ${exam_entry})
			file(GLOB exam_sources "${exam_entry}/*.cpp")
			
			get_filename_component(exam_name "${exam_entry}" NAME)
			
			add_executable(${PROJECT_NAME}_exam_${exam_name} EXCLUDE_FROM_ALL
				${exam_sources}
			)
			target_link_libraries(${PROJECT_NAME}_exam_${exam_name}
				PRIVATE
					${PROJECT_NAME}_dep_private
					${PROJECT_NAME}_dep_export
					${PROJECT_NAME}_lib
					${PROJECT_NAME}_build_private
					${PROJECT_NAME}_build_export
			)
			set_target_properties(${PROJECT_NAME}_exam_${exam_name} PROPERTIES OUTPUT_NAME ${exam_name})
			set_target_properties("${PROJECT_NAME}_exam_${exam_name}" PROPERTIES
				RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debugs/examples"
				RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/releases/examples"
			)
			list(APPEND install_example_target_list ${PROJECT_NAME}_exam_${exam_name})
		elseif(NOT IS_DIRECTORY ${exam_entry})
			get_filename_component(exam_name "${exam_entry}" NAME_WE)
			message("exam_name: ${exam_name}")
			
			add_executable(${PROJECT_NAME}_exam_${exam_name} EXCLUDE_FROM_ALL
				${exam_entry}
			)
			target_link_libraries(${PROJECT_NAME}_exam_${exam_name}
				PRIVATE
					${PROJECT_NAME}_dep_private
					${PROJECT_NAME}_dep_export
					${PROJECT_NAME}_lib
					${PROJECT_NAME}_build_private
					${PROJECT_NAME}_build_export
			)
			set_target_properties(${PROJECT_NAME}_exam_${exam_name} PROPERTIES OUTPUT_NAME "${exam_name}")
			set_target_properties("${PROJECT_NAME}_exam_${exam_name}" PROPERTIES
				RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/debugs/examples"
				RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/releases/examples"
			)
			list(APPEND install_example_target_list ${PROJECT_NAME}_exam_${exam_name})
		endif()
	endforeach()
	message(STATUS "install_example_target_list: ${install_example_target_list}")
	add_custom_target(${PROJECT_NAME}_examples)
	add_dependencies(${PROJECT_NAME}_examples ${install_example_target_list})
endblock()