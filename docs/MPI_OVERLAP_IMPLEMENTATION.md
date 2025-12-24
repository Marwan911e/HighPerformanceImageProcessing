# MPI Computation-Communication Overlap Implementation

## Overview

This document describes the implementation of computation-communication overlap in the MPI parallelization, a critical optimization technique that hides communication latency behind useful computation.

---

## Why Overlap Matters

In distributed-memory systems, communication between processes can be expensive. By overlapping communication with computation, we can:

1. **Hide Latency**: CPU performs useful work while waiting for network data
2. **Improve Efficiency**: Better utilization of both CPU and network resources
3. **Increase Speedup**: Closer to ideal parallel performance

---

## Architecture

### 1. Data Structures

**BoundaryExchange Struct** (`include/mpi_utils.h:8-15`)
```cpp
struct BoundaryExchange {
    Image extended;              // Image with space for halos
    std::vector<MPI_Request> requests;  // Non-blocking MPI requests
    int rank;
    int size;
    int haloSize;
};
```

This structure encapsulates:
- Extended image buffer with ghost cell space
- MPI request handles for non-blocking operations
- Context information (rank, size, halo size)

---

### 2. Core Functions

#### startBoundaryExchange()
**Location**: `src/mpi_utils.cpp:227-288`

**Purpose**: Initiate non-blocking halo exchange

**Flow**:
1. Allocate extended image with halo space
2. Copy local data to center of extended image
3. Start MPI_Isend to send boundary rows to neighbors
4. Start MPI_Irecv to receive halo rows from neighbors
5. **Return immediately** (don't wait)

**Key Point**: Returns control to caller while communication is in progress

```cpp
BoundaryExchange exchange = startBoundaryExchange(localChunk, rank, size, haloSize);
// Communication is now happening in background
// Can do computation here!
```

---

#### waitBoundaryExchange()
**Location**: `src/mpi_utils.cpp:290-297`

**Purpose**: Wait for halo exchange to complete

**Flow**:
1. Call MPI_Waitall on all pending requests
2. Ensures all halo data has arrived
3. Safe to access halo regions after this call

```cpp
waitBoundaryExchange(exchange);
// All halos are now available
```

---

#### getInnerRegion()
**Location**: `src/mpi_utils.cpp:299-311`

**Purpose**: Identify rows that don't need halo data

**Logic**:
- Inner region = rows that are at least `haloSize` away from boundaries
- These pixels can be computed using only locally available data
- Can be processed while waiting for halos

```
Extended Image Layout:
┌─────────────────┐
│   Top Halo      │ ← From rank-1 (needs communication)
├─────────────────┤
│   Top Border    │ ← Needs halo data
├─────────────────┤
│                 │
│  INNER REGION   │ ← Can compute immediately!
│                 │
├─────────────────┤
│  Bottom Border  │ ← Needs halo data
├─────────────────┤
│  Bottom Halo    │ ← From rank+1 (needs communication)
└─────────────────┘
```

---

#### getBorderRegions()
**Location**: `src/mpi_utils.cpp:313-328`

**Purpose**: Identify rows that need halo data

**Logic**:
- Top border = first `haloSize` rows of local data
- Bottom border = last `haloSize` rows of local data
- Must wait for halos before processing these

---

## Usage Pattern

### Example: Box Blur Filter

**File**: `src/filters_mpi.cpp:44-100`

```cpp
Image boxBlur(const Image& localChunk, int kernelSize, int rank, int size) {
    int radius = kernelSize / 2;
    int haloSize = radius;
    
    // PHASE 1: Start non-blocking communication
    BoundaryExchange exchange = startBoundaryExchange(localChunk, rank, size, haloSize);
    // ⚡ Communication is now in progress
    
    Image filtered = exchange.extended.createSimilar();
    
    // PHASE 2: Compute inner region (OVERLAPS with communication)
    int innerStart, innerEnd;
    getInnerRegion(localChunk.getHeight(), haloSize, rank, size, innerStart, innerEnd);
    if (innerStart < innerEnd) {
        applyBoxBlurRegion(exchange.extended, filtered, kernelSize, innerStart, innerEnd);
        // ⚡ CPU is busy while network transfers data
    }
    
    // PHASE 3: Wait for halos (synchronization point)
    waitBoundaryExchange(exchange);
    // ✓ All halo data has arrived
    
    // PHASE 4: Compute border regions (uses halo data)
    int topStart, topEnd, bottomStart, bottomEnd;
    getBorderRegions(localChunk.getHeight(), haloSize, rank, size, 
                     topStart, topEnd, bottomStart, bottomEnd);
    if (topStart < topEnd) {
        applyBoxBlurRegion(exchange.extended, filtered, kernelSize, topStart, topEnd);
    }
    if (bottomStart < bottomEnd) {
        applyBoxBlurRegion(exchange.extended, filtered, kernelSize, bottomStart, bottomEnd);
    }
    
    return removeHalo(filtered, rank, size, haloSize);
}
```

---

## Timeline Comparison

### Without Overlap (Old Approach)
```
Rank 0:  |--Send--|--Recv--||--------Compute--------|
Rank 1:  |--Send--|--Recv--||--------Compute--------|
         ^                  ^
         Start comm         Start compute
         
Total time = Communication + Computation
```

### With Overlap (New Approach)
```
Rank 0:  |--Send--|--Recv--
         |         |--------Compute Inner---|
         |                 Wait |--Border--|
                                ^
Rank 1:  |--Send--|--Recv--     Synchronize
         |         |--------Compute Inner---|
         |                 Wait |--Border--|

Total time ≈ max(Communication, Inner Computation) + Border Computation
```

**Speedup**: Communication time is hidden by inner region computation!

---

## Implementation in All Filters

All MPI convolution filters use this pattern:

### Filters (src/filters_mpi.cpp)
- ✅ boxBlur
- ✅ gaussianBlur
- ✅ medianFilter
- ✅ bilateralFilter

### Edge Detection (src/edge_detection_mpi.cpp)
- ✅ sobel
- ✅ sharpen
- ✅ prewitt
- ✅ laplacian
- ⚠️ canny (uses overlap but has complex hysteresis step)

---

## Performance Considerations

### When Overlap is Most Effective

1. **Large images**: More inner region pixels → more computation to overlap
2. **Small halos**: Less data to transfer → faster communication
3. **Complex filters**: Computation-heavy operations benefit most
4. **High network latency**: More time to hide

### Optimal Conditions

- Inner region should have enough work to cover communication time
- For 3x3 kernels (haloSize=1), inner region = height - 2 rows per process
- Larger kernels → larger halos → smaller inner region

---

## Verification

To verify overlap is working:

1. **Code inspection**: Check that computation happens between `start` and `wait`
2. **Profiling**: Use MPI profilers to see computation during communication
3. **Timing**: Compare performance with/without overlap

---

## Best Practices

1. **Always use minimal halos**: `haloSize = kernelRadius`
2. **Maximize inner region**: More overlap opportunity
3. **Region-specific computation**: Don't recompute the entire image
4. **Handle edge cases**: Check bounds for small images/many processes

---

## Future Optimizations

Possible enhancements:

1. **Double buffering**: Prepare next operation while current one completes
2. **Pipelined stages**: Overlap multiple operations
3. **GPU overlap**: Similar pattern for GPU communication
4. **Adaptive strategies**: Choose overlap based on image size and filter complexity

---

## References

- MPI-3.1 Standard: Non-blocking Communication
- "Using MPI" by Gropp, Lusk, and Skjellum
- HPC best practices for stencil computations

---

## Summary

The computation-communication overlap implementation:
- ✅ Uses non-blocking MPI primitives (Isend/Irecv)
- ✅ Splits computation into independent regions
- ✅ Hides communication latency behind useful work
- ✅ Follows HPC best practices
- ✅ Provides clear, reusable pattern for all filters

This is a fundamental optimization for achieving high performance in distributed-memory parallel image processing.

