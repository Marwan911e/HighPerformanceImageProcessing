# Quick Fix for Cluster

You have local changes blocking the pull. Run these commands on the cluster:

```bash
# Option 1: Stash your changes, pull, then reapply
git stash
git pull origin cuda
git stash pop

# Option 2: Discard local changes and pull (if you don't need them)
git checkout -- build_cluster_aast.sh
git pull origin cuda
```

After pulling, the build script should work. If it still shows the c++17 error, the version detection might need adjustment.

## Manual Fix (if version detection still fails)

Edit `build_cluster_aast.sh` and change line 42 from:
```bash
submit.nvcc -c ../src/cuda_kernels.cu -o cuda_kernels.o -I../include -I../lib -arch=sm_70 -O3 -std=$CPP_STD
```

To:
```bash
submit.nvcc -c ../src/cuda_kernels.cu -o cuda_kernels.o -I../include -I../lib -arch=sm_70 -O3 -std=c++14
```

And change all the g++ lines from `-std=$CPP_STD` to `-std=c++14`

