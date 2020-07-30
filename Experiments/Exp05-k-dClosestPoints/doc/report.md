

```bash
# karl @ karl in ~/Coding/Cuda-Programming/Experiments/Exp05-k-dClosestPoints/baseline(sequential)/src [14:45:39] 
$ cudamake
-- The C compiler identification is GNU 4.8.5
-- The CXX compiler identification is GNU 4.8.5
-- Check for working C compiler: /usr/bin/gcc-4.8
-- Check for working C compiler: /usr/bin/gcc-4.8 -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /usr/bin/g++-4.8
-- Check for working CXX compiler: /usr/bin/g++-4.8 -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Looking for pthread.h
-- Looking for pthread.h - found
-- Looking for pthread_create
-- Looking for pthread_create - not found
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - found
-- Found Threads: TRUE  
-- Configuring done
-- Generating done
-- Build files have been written to: /home/karl/Coding/Cuda-Programming/Experiments/Exp05-k-dClosestPoints/baseline(sequential)/build
[ 33%] Building NVCC (Device) object CMakeFiles/main.dir/main_generated_main.cu.o
[ 66%] Building NVCC (Device) object CMakeFiles/main.dir/main_generated_core.cu.o
Scanning dependencies of target main
[100%] Linking CXX executable main
[100%] Built target main



```

baseline

```bash
Small sample 0:
---
Search points:
- [ 0.40 0.60 0.60 ]
Reference points:
- 0: [ 0.10 0.50 0.80 ]
- 1: [ 1.00 1.00 0.50 ]
Results:
- 0 (0.374)
cudaCallback: 0.000 ms


Small sample 1:
---
Search points:
- [ 0.60 0.90 0.40 ]
- [ 0.80 0.70 0.10 ]
Reference points:
- 0: [ 0.10 0.60 0.70 ]
- 1: [ 0.70 0.80 0.80 ]
- 2: [ 0.80 0.80 0.60 ]
- 3: [ 0.70 0.30 0.70 ]
- 4: [ 0.10 0.50 0.30 ]
- 5: [ 0.60 0.90 0.80 ]
- 6: [ 0.20 1.00 0.30 ]
- 7: [ 0.00 1.00 0.30 ]
Results:
- 2 (0.300)
- 2 (0.510)
cudaCallback: 0.001 ms


Sample 2:
---
cudaCallback: 0.013 ms


Sample 3:
---
cudaCallback: 0.818 ms


Sample 4:
---
cudaCallback: 3.737 ms


Sample 5:
---
cudaCallback: 14.214 ms


Sample 6:
---
cudaCallback: 805.130 ms


Sample 7:
---
cudaCallback: 3643.300 ms
```





referenceParallel

```bash
Small sample 0:
---
Search points:
- [ 0.40 0.60 0.60 ]
Reference points:
- 0: [ 0.10 0.50 0.80 ]
- 1: [ 1.00 1.00 0.50 ]
Results:
- 0 (0.374)
cudaCallback: 69.444 ms


Small sample 1:
---
Search points:
- [ 0.60 0.90 0.40 ]
- [ 0.80 0.70 0.10 ]
Reference points:
- 0: [ 0.10 0.60 0.70 ]
- 1: [ 0.70 0.80 0.80 ]
- 2: [ 0.80 0.80 0.60 ]
- 3: [ 0.70 0.30 0.70 ]
- 4: [ 0.10 0.50 0.30 ]
- 5: [ 0.60 0.90 0.80 ]
- 6: [ 0.20 1.00 0.30 ]
- 7: [ 0.00 1.00 0.30 ]
Results:
- 2 (0.300)
- 2 (0.510)
cudaCallback: 0.185 ms


Sample 2:
---
cudaCallback: 0.166 ms


Sample 3:
---
cudaCallback: 0.859 ms


Sample 4:
---
cudaCallback: 2.039 ms


Sample 5:
---
cudaCallback: 24.153 ms


Sample 6:
---
cudaCallback: 282.930 ms


Sample 7:
---
cudaCallback: 372.262 ms
```



searchParallel

```bash
Small sample 0:
---
Search points:
- [ 0.40 0.60 0.60 ]
Reference points:
- 0: [ 0.10 0.50 0.80 ]
- 1: [ 1.00 1.00 0.50 ]
Results:
- 0 (0.374)
cudaCallback: 74.530 ms


Small sample 1:
---
Search points:
- [ 0.60 0.90 0.40 ]
- [ 0.80 0.70 0.10 ]
Reference points:
- 0: [ 0.10 0.60 0.70 ]
- 1: [ 0.70 0.80 0.80 ]
- 2: [ 0.80 0.80 0.60 ]
- 3: [ 0.70 0.30 0.70 ]
- 4: [ 0.10 0.50 0.30 ]
- 5: [ 0.60 0.90 0.80 ]
- 6: [ 0.20 1.00 0.30 ]
- 7: [ 0.00 1.00 0.30 ]
Results:
- 2 (0.300)
- 2 (0.510)
cudaCallback: 0.166 ms


Sample 2:
---
cudaCallback: 0.589 ms


Sample 3:
---
cudaCallback: 31.381 ms


Sample 4:
---
cudaCallback: 51.396 ms


Sample 5:
---
cudaCallback: 0.749 ms


Sample 6:
---
cudaCallback: 36.112 ms


Sample 7:
---
cudaCallback: 396.905 ms

```



combinedParallel

```bash
Small sample 0:
---
Search points:
- [ 0.40 0.60 0.60 ]
Reference points:
- 0: [ 0.10 0.50 0.80 ]
- 1: [ 1.00 1.00 0.50 ]
Results:
- 0 (0.374)
cudaCallback: 85.060 ms


Small sample 1:
---
Search points:
- [ 0.60 0.90 0.40 ]
- [ 0.80 0.70 0.10 ]
Reference points:
- 0: [ 0.10 0.60 0.70 ]
- 1: [ 0.70 0.80 0.80 ]
- 2: [ 0.80 0.80 0.60 ]
- 3: [ 0.70 0.30 0.70 ]
- 4: [ 0.10 0.50 0.30 ]
- 5: [ 0.60 0.90 0.80 ]
- 6: [ 0.20 1.00 0.30 ]
- 7: [ 0.00 1.00 0.30 ]
Results:
- 2 (0.300)
- 2 (0.510)
cudaCallback: 0.234 ms


Sample 2:
---
cudaCallback: 0.202 ms


Sample 3:
---
cudaCallback: 0.905 ms


Sample 4:
---
cudaCallback: 1.501 ms


Sample 5:
---
cudaCallback: 0.729 ms


Sample 6:
---
cudaCallback: 44.789 ms


Sample 7:
---
cudaCallback: 370.451 ms
```



dynamicParallel

```bash
Small sample 0:
---
Search points:
- [ 0.40 0.60 0.60 ]
Reference points:
- 0: [ 0.10 0.50 0.80 ]
- 1: [ 1.00 1.00 0.50 ]
Results:
- 0 (0.374)
cudaCallback: 105.276 ms


Small sample 1:
---
Search points:
- [ 0.60 0.90 0.40 ]
- [ 0.80 0.70 0.10 ]
Reference points:
- 0: [ 0.10 0.60 0.70 ]
- 1: [ 0.70 0.80 0.80 ]
- 2: [ 0.80 0.80 0.60 ]
- 3: [ 0.70 0.30 0.70 ]
- 4: [ 0.10 0.50 0.30 ]
- 5: [ 0.60 0.90 0.80 ]
- 6: [ 0.20 1.00 0.30 ]
- 7: [ 0.00 1.00 0.30 ]
Results:
- 0 (0.656)
- 0 (0.927)
cudaCallback: 0.234 ms


Sample 2:
---
cudaCallback: 0.230 ms


Sample 3:
---
cudaCallback: 2.713 ms


Sample 4:
---
cudaCallback: 3.515 ms


Sample 5:
---
cudaCallback: 4.339 ms


Sample 6:
---
cudaCallback: 19.648 ms


Sample 7:
---
cudaCallback: 128.083 ms
```



kdTree

```bash
Small sample 0:
---
Search points:
- [ 0.40 0.60 0.60 ]
Reference points:
- 0: [ 0.10 0.50 0.80 ]
- 1: [ 1.00 1.00 0.50 ]
Results:
- 0 (0.374)
cudaCallback: 0.004 ms


Small sample 1:
---
Search points:
- [ 0.60 0.90 0.40 ]
- [ 0.80 0.70 0.10 ]
Reference points:
- 0: [ 0.10 0.60 0.70 ]
- 1: [ 0.70 0.80 0.80 ]
- 2: [ 0.80 0.80 0.60 ]
- 3: [ 0.70 0.30 0.70 ]
- 4: [ 0.10 0.50 0.30 ]
- 5: [ 0.60 0.90 0.80 ]
- 6: [ 0.20 1.00 0.30 ]
- 7: [ 0.00 1.00 0.30 ]
Results:
- 2 (0.300)
- 2 (0.510)
cudaCallback: 0.007 ms


Sample 2:
---
cudaCallback: 1.500 ms


Sample 3:
---
cudaCallback: 235.028 ms


Sample 4:
---
cudaCallback: 323.271 ms


Sample 5:
---
cudaCallback: 3.809 ms


Sample 6:
---
cudaCallback: 264.501 ms


Sample 7:
---
cudaCallback: 903.987 ms
```

