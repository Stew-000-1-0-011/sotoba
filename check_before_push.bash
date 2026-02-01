#!/bin/bash

# リポジトリのルートへ移動（念のため）
ROOT_DIR=$(git rev-parse --show-toplevel)
cd "$ROOT_DIR" || exit 1

. source_setvars.bash

# フォーマットのチェック
echo "Checking code formatting..." >&2
# 対象の拡張子 (.c, .cpp, .h, .hpp など)
# git ls-filesを使うことで、git管理下のファイルのみを対象にします
FILES=$(git ls-files | grep -E '\.(c|cpp|h|hpp)$')

if [ -n "$FILES" ]; then
	# xargsを使ってファイルリストを渡し、dry-run(変更せず確認)とWerror(警告をエラー扱い)を実行
	# 注意: Clang 10以上で --dry-run --Werror が使用可能です
	echo "$FILES" | xargs clang-format --dry-run --Werror >&2
	
	if [ $? -ne 0 ]; then
		echo "Error: Code formatting issues detected." >&2
		echo "Please run clang-format to fix them." >&2
		exit 1
	fi
else
	echo "No source files found to format." >&2
fi

# 1. ビルドの実行
echo "Running build.bash_template..." >&2
if ! bash build.bash_template; then
	echo "Error: Build failed." >&2
	exit 1
fi

# 2. テストの実行 (ctest)
# ここにctestを実行したいディレクトリのパスを列挙してください
TEST_DIRS=(
	"clang_build/Debug/"
	"clang_build/Release/"
	# "gcc_build/Debug/"
	# "gcc_build/Release/"
	# "icpx_build/Debug/"
	# "icpx_build/Release/"
)

echo "Running tests..." >&2
for dir in "${TEST_DIRS[@]}"; do
	if [ -d "$dir" ]; then
		echo "Testing in: $dir" >&2
		# ディレクトリ移動して実行
		pushd "$dir" > /dev/null || exit 1
		
		# ctest実行。失敗したらループを抜けて終了
		if ! ctest; then
			echo "Error: ctest failed in $dir" >&2
			popd > /dev/null || exit 1
			exit 1
		fi
		
		# 元のディレクトリに戻る
		popd > /dev/null || exit 1
	else
		echo "Warning: Directory not found: $dir" >&2
	fi
done

echo "All checks passed. Proceeding with push." >&2
exit 0
