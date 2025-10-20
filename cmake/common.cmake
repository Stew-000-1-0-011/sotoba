# --- 空のライブラリ定義 ---
# 普段は変える必要ない

add_library(${PROJECT_NAME}_build_common_private INTERFACE EXCLUDE_FROM_ALL)
target_compile_options(${PROJECT_NAME}_build_common_private
	INTERFACE
		-march=native
		-fno-omit-frame-pointer
		-fvisibility=hidden
		-Wall
		-Wextra
		-Wpedantic
)


add_library(${PROJECT_NAME}_build_debug_private INTERFACE EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME}_build_debug_private
	INTERFACE
		${PROJECT_NAME}_build_common_private
)
target_compile_definitions(${PROJECT_NAME}_build_debug_private
	INTERFACE
		${PROJECT_NAME}_DEBUG
)
target_compile_options(${PROJECT_NAME}_build_debug_private
	INTERFACE
		-g
		-O0
)
target_link_options(${PROJECT_NAME}_build_debug_private
	INTERFACE
		-g
)


add_library(${PROJECT_NAME}_build_debug_export INTERFACE EXCLUDE_FROM_ALL)
target_compile_options(${PROJECT_NAME}_build_debug_export
	INTERFACE
		-fsanitize=address
		-fsanitize=undefined
)
target_link_options(${PROJECT_NAME}_build_debug_export
	INTERFACE
		-fsanitize=address
		-fsanitize=undefined
)


add_library(${PROJECT_NAME}_build_release_private INTERFACE EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME}_build_release_private
	INTERFACE
		${PROJECT_NAME}_build_common_private
)
target_compile_definitions(${PROJECT_NAME}_build_release_private
	INTERFACE
		${PROJECT_NAME}_RELEASE
)
target_compile_options(${PROJECT_NAME}_build_release_private
	INTERFACE
		${RELEASE_OPTIMIZE}
		# -flto
)
# target_link_options(${PROJECT_NAME}_build_release_private
# 	INTERFACE
# 		-flto
# )


add_library(${PROJECT_NAME}_build_release_export INTERFACE EXCLUDE_FROM_ALL)


add_library(${PROJECT_NAME}_build_private INTERFACE EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME}_build_private
	INTERFACE
		${PROJECT_NAME}_build_$<IF:$<CONFIG:Debug>,debug,release>_private
)

add_library(${PROJECT_NAME}_build_export INTERFACE EXCLUDE_FROM_ALL)
target_link_libraries(${PROJECT_NAME}_build_export
	INTERFACE
		${PROJECT_NAME}_build_$<IF:$<CONFIG:Debug>,debug,release>_export
)