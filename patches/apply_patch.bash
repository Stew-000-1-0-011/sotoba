#!/bin/bash
set -e  # エラーがあれば即停止

PATCH_FILE="$1"
MARKER_FILE=".patch_applied"

# マーカーファイルがあれば何もしないで正常終了
if [ -f "$MARKER_FILE" ]; then
    echo "Patch already applied. Skipping."
    exit 0
fi

# パッチ適用とマーカー作成
echo "Applying patch: $PATCH_FILE"
git apply "$PATCH_FILE"
touch "$MARKER_FILE"