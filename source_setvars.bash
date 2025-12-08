#!/bin/bash
# 冪等性を持たせるためにこうしている(pre-pushなどに必要)

echo "source oneapi/setvars.sh if never done..." >&2

# ONEAPI_ROOT が空の場合のみ実行する
if [ -z "${ONEAPI_ROOT:-}" ]; then
	# 画面サイズのエラー回避
	export COLUMNS=80
	export LINES=24
	
	# 出力を捨てつつ実行
	. /opt/intel/oneapi/setvars.sh
	# . /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
fi
