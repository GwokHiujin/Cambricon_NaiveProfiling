#!/bin/bash
set -e

source env.sh

check_deb_package() {
    if [ -z "$(dpkg -l | grep ${1})" ]; then
        echo "Please sudo apt install ${1}"
        exit -1
    fi
}

check_rpm_package() {
    if [ -z "$(rpm -qa | grep ${1})" ]; then
        echo "Please sudo yum install ${1}"
        exit -1
    fi
}

# check dependency for build
if [ -f "/etc/os-release" ]; then
    source /etc/os-release
    if [ ${ID} == "ubuntu" ] || [ ${ID} == "debian" ]; then
        check_deb_package cmake
        CMAKE=cmake
    elif [ ${ID} == "centos" ]; then
        if [ ${VERSION_ID} == "8" ]; then
            check_rpm_package cmake
            CMAKE=cmake
        else
            check_rpm_package cmake3
            CMAKE=cmake3
	fi
    elif [ ${ID} == "kylin" ]; then
        check_rpm_package cmake
        CMAKE=cmake
    else
	echo "Cmake not support on this os!"
	exit -1
    fi
fi

export WORKSPACE=${PWD}

################################################################################
# NOTE: NEUWARE_ROUNDING_MODE
#
# This env variable used for setting the hardware computing rounding mode
# The related computing unit is ALU, NFU, WFU.
# The cncc will set NEUWARE_ROUNDING_MODE=5 defaultly.
#
# [NULL,5] - Rounding to nearest even (= CPU/GPU)
# [0]      - Rounding to to zero
# [1]      - Rounding to positive Inf
# [2]      - Rounding up
# [3]      - Rounding down
# [4]      - Rounding to negative Inf
################################################################################
export NEUWARE_ROUNDING_MODE=0

usage () {
    echo "USAGE: build.sh <options>"
    echo
    echo "       If need specify neuware path, please:"
    echo "       Firt, export NEUWARE_HOME where neuare installed"
    echo "         export NEUWARE_HOME=/path/of/your/neuware"
    echo "       Second, export TOOLCHAIN_ROOT if cross-compilation for aarch64-linux-gnu"
    echo "         export TOOLCHAIN_ROOT=/path/to/cross_toolchains"
    echo
    echo "OPTIONS:"
    echo "      -h, --help                     Print usage"
    echo "      <null>                         If no --mluxxx specified, default option is --compute_20"
    echo "      --1m20                         Build for target product 1M20:   __BANG_ARCH__ = 220"
    echo "                                                                      __MLU_NRAM_SIZE__ = 512KB"
    echo "                                                                      __MLU_WRAM_SIZE__ = 512KB"
    echo "                                                                      __MLU_SRAM_SIZE__ = 0KB"
    echo "                                                                      cncc --bang-mlu-arch=tp_220,  cnas --mlu-arch tp_220"
    echo "      --mlu220                       Build for target product MLU220: __BANG_ARCH__ = 220"
    echo "                                                                      __MLU_NRAM_SIZE__ = 512KB"
    echo "                                                                      __MLU_WRAM_SIZE__ = 512KB"
    echo "                                                                      __MLU_SRAM_SIZE__ = 2048KB"
    echo "                                                                      cncc --bang-mlu-arch=mtp_220, cnas --mlu-arch mtp_220"
    echo "      --1m70                         Build for target product 1M70:   __BANG_ARCH__ = 270"
    echo "                                                                      __MLU_NRAM_SIZE__ = 512KB"
    echo "                                                                      __MLU_WRAM_SIZE__ = 1024KB"
    echo "                                                                      __MLU_SRAM_SIZE__ = 0KB"
    echo "                                                                      cncc --bang-mlu-arch=tp_270,  cnas --mlu-arch tp_270"
    echo "      --mlu270                       Build for target product MLU270: __BANG_ARCH__ = 270"
    echo "                                                                      __MLU_NRAM_SIZE__ = 512KB"
    echo "                                                                      __MLU_WRAM_SIZE__ = 1024KB"
    echo "                                                                      __MLU_SRAM_SIZE__ = 2048KB"
    echo "                                                                      cncc --bang-mlu-arch=mtp_270, cnas --mlu-arch mtp_270"
    echo "      --mlu290                       Build for target product MLU290: __BANG_ARCH__ = 290"
    echo "                                                                      __MLU_NRAM_SIZE__ = 512KB"
    echo "                                                                      __MLU_WRAM_SIZE__ = 512KB"
    echo "                                                                      __MLU_SRAM_SIZE__ = 2048KB"
    echo "                                                                      cncc --bang-mlu-arch=mtp_290, cnas --mlu-arch mtp_290"
    echo "      --compute_20                   Build for target product MLU220/MLU270/MLU290 with cnfabin, which automatically select target mlu arch."
    echo "                                                                      cncc --bang-arch=compute_20, equal to combination of options below:"
    echo "                                                                      cncc --bang-mlu-arch=mtp_220 --bang-mlu-arch=mtp_270 --bang-mlu-arch=mtp_290"
    echo "      --ce3226                       Build for target product CE3226: __BANG_ARCH__ = 322"
    echo "                                                                      __MLU_NRAM_SIZE__ = 768KB"
    echo "                                                                      __MLU_WRAM_SIZE__ = 1024KB"
    echo "                                                                      __MLU_SRAM_SIZE__ = 0KB"
    echo "                                                                      cncc --bang-mlu-arch=tp_322, cnas --mlu-arch tp_322"
    echo "      --mlu370                       Build for target product MLU370: __BANG_ARCH__ = 372"
    echo "                                                                      __MLU_NRAM_SIZE__ = 768KB"
    echo "                                                                      __MLU_WRAM_SIZE__ = 1024KB"
    echo "                                                                      __MLU_SRAM_SIZE__ = 2048KB"
    echo "                                                                      cncc --bang-mlu-arch=mtp_372, cnas --mlu-arch mtp_372"
    echo "      --compute_30                   Build for target product MLU370/MLU3xx... with cnfabin, which automatically select target mlu arch."
    echo "                                                                      cncc --bang-arch=compute_30, equal to combination of options below:"
    echo "                                                                      cncc --bang-mlu-arch=mtp_372 ..."
    echo "      --mlu590                       Build for target product MLU590: __BANG_ARCH__ = 592"
    echo "                                                                      __MLU_NRAM_SIZE__ = 512KB"
    echo "                                                                      __MLU_WRAM_SIZE__ = 512KB"
    echo "                                                                      __MLU_SRAM_SIZE__ = 2048KB"
    echo "                                                                      cncc --bang-mlu-arch=mtp_592, cnas --mlu-arch mtp_592"
    echo "      --aarch64                      Cross compilation build for edge device, target cpu arch is aarch64-linux-gnu."
    echo "      -d, --debug                    Build test case with debug mode"
    echo
}

# Handle build mode
BUILDDIR="./build"
if [ ! -d "$BUILDDIR" ]; then
  mkdir "$BUILDDIR"
fi

# Detect arch of host cpu
TARGET_CPU_ARCH=${TARGET_CPU_ARCH:-$(uname -m)-linux-gnu}

# Default target mlu arch is compute_20 which support running on MLU220/MLU270/MLU290 device.
TARGET_MLU_ARCH="compute_20"

if [ $# != 0 ]; then
  BUILD_MODE="release"
  while [ $# != 0 ]; do
    case "$1" in
      -h | --help)
          usage
          exit 0
          ;;
      --1m10)
          TARGET_MLU_ARCH="tp_210"
          shift
          ;;
      --1m20)
          TARGET_MLU_ARCH="tp_220"
          shift
          ;;
      --1m70)
          TARGET_MLU_ARCH="tp_270"
          shift
          ;;
      --mlu220)
          TARGET_MLU_ARCH="mtp_220"
          shift
          ;;
      --mlu270)
          TARGET_MLU_ARCH="mtp_270"
          shift
          ;;
      --mlu290)
          TARGET_MLU_ARCH="mtp_290"
          shift
          ;;
      --compute_20)
          TARGET_MLU_ARCH="compute_20"
          shift
          ;;
      --ce3226)
          TARGET_MLU_ARCH="tp_322"
          TARGET_CPU_ARCH="aarch64-linux-gnu"
          shift
          ;;
      --mlu370)
          TARGET_MLU_ARCH="mtp_372"
          shift
          ;;
      --compute_30)
          TARGET_MLU_ARCH="compute_30"
          shift
          ;;
      --aarch64)
          TARGET_CPU_ARCH="aarch64-linux-gnu"
          shift
          ;;
      --mlu590)
          TARGET_MLU_ARCH="mtp_592"
          shift
          ;;
      -d | --debug)
          BUILD_MODE="debug"
          shift
          ;;
      *)
          echo "unknown options ${1}, use -h or --help"
          exit -1;
          ;;
    esac
  done
else
  BUILD_MODE="release"
fi

if [ ${TARGET_CPU_ARCH} != "$(uname -m)-linux-gnu" ]; then
  CXX_COMPILER="${TOOLCHAIN_ROOT}/bin/aarch64-linux-gnu-g++"
  C_COMPILER="${TOOLCHAIN_ROOT}/bin/aarch64-linux-gnu-gcc"
  SYSROOT="$(find ${TOOLCHAIN_ROOT} -name "libc" -type d)"
else
  CXX_COMPILER="$(which g++)"
  C_COMPILER="$(which gcc)"
fi

# Build samples for target mlu arch
pushd ${BUILDDIR}
  rm -rf *
  echo "-- Build in ${BUILD_MODE} mode"
  ${CMAKE} -DCMAKE_BUILD_TYPE=${BUILD_MODE} \
           -DCMAKE_SYSROOT=${SYSROOT} \
           -DTOOLCHAIN_ROOT=${TOOLCHAIN_ROOT} \
           -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
           -DCMAKE_C_COMPILER=${C_COMPILER} \
           -DNEUWARE_HOME=${NEUWARE_HOME} \
           -DTARGET_MLU_ARCH=${TARGET_MLU_ARCH} \
           -DTARGET_CPU_ARCH=${TARGET_CPU_ARCH} \
           ..
  make -j32
  ln -sf ../input_data
popd
