#!/bin/bash
# This script extracts necessary files from the official NVIDIA SDK:
# * public header files of the NGX SDK
# * Windows DLL implementation of the various features (nvngx_*.dll)

set -ex

if [ $# -ne 1 ]; then
  echo "Usage: $0 path/to/NGX_SDK_EA1.1.exe" 1>&2
  exit 1
fi

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
NGX_INCLUDE_DIR="$SCRIPT_DIR/include/ngx"
NGX_DLL_DIR="$SCRIPT_DIR/ngx_dlls"
mkdir -p "$NGX_INCLUDE_DIR"
mkdir -p "$NGX_DLL_DIR"

TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

pushd $TMP_DIR
7z x "$1" "NGX SDK/"
mv NGX\ SDK/*.h "$NGX_INCLUDE_DIR"
mv NGX\ SDK/nvngx_*.dll "$NGX_DLL_DIR"
