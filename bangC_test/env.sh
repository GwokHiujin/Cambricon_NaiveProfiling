if [ -z "${NEUWARE_HOME}" ]; then
  export NEUWARE_HOME=/usr/local/neuware
fi
if [ -z "${TOOLCHAIN_ROOT}" ]; then
  export TOOLCHAIN_ROOT="/path/to/your/cross_toolchains"
fi

export PATH=${NEUWARE_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib:${LD_LIBRARY_PATH}
