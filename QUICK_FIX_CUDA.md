# Quick Fix: CUDA Path Not Found

If the build script can't find CUDA automatically, set it manually:

```bash
export CUDA_HOME=/usr/local/cuda-10.0
./build_cluster_aast.sh
```

Or check if the directory exists:
```bash
ls -la /usr/local/cuda-10.0/include/cuda_runtime.h
```

If it doesn't exist, try:
```bash
find /usr /opt -name "cuda_runtime.h" 2>/dev/null | grep -v python | grep -v site-packages
```

Then set CUDA_HOME to the directory containing the `include` folder.

