# MPI Requirements Analysis

## Summary
**Status: ❌ Does NOT fully match requirements**

The current implementation has the basic structure for MPI parallelization but is missing several critical requirements.

---

## Detailed Requirement Check

### ✅ Requirement 1: Partition image into contiguous chunks (row-wise) using MPI_Scatterv/MPI_Gatherv
**Status: ❌ NOT IMPLEMENTED**

**Current Implementation:**
- Uses `MPI_Send`/`MPI_Recv` in loops (lines 77-91 in `mpi_utils.cpp`)
- Does NOT use `MPI_Scatterv` or `MPI_Gatherv`
- Handles variable-sized chunks correctly but inefficiently

**What's Needed:**
```cpp
// Should use MPI_Scatterv for distribution
MPI_Scatterv(sendbuf, sendcounts, displs, MPI_BYTE, 
             recvbuf, recvcount, MPI_BYTE, 0, MPI_COMM_WORLD);

// Should use MPI_Gatherv for gathering
MPI_Gatherv(sendbuf, sendcount, MPI_BYTE,
            recvbuf, recvcounts, displs, MPI_BYTE, 0, MPI_COMM_WORLD);
```

---

### ❌ Requirement 2: Halo/Ghost rows for convolution filters
**Status: ❌ INCOMPLETE**

**Current Implementation:**
- `exchangeBoundaries()` function exists (lines 148-176 in `mpi_utils.cpp`) but is incomplete
- Only sends boundaries, does NOT receive them
- Does NOT extend image with ghost cells
- **NOT USED** anywhere in the codebase
- Convolution filters (blur, edge detection) are called directly without halo exchange

**What's Needed:**
1. Complete `exchangeBoundaries()` to:
   - Receive halo rows from neighbors
   - Extend local image with ghost cells
   - Keep halo size minimal (kernel radius)
2. Use halo exchange before convolution operations
3. Compute inner region while waiting for halo data

---

### ❌ Requirement 3: Non-blocking communication (MPI_Isend, MPI_Irecv)
**Status: ❌ NOT IMPLEMENTED**

**Current Implementation:**
- All communication is **blocking** (`MPI_Send`, `MPI_Recv`)
- No overlap of communication with computation
- No use of `MPI_Isend`, `MPI_Irecv`, or `MPI_Waitall`

**What's Needed:**
```cpp
// Start non-blocking receives
MPI_Irecv(topHalo, ..., rank-1, ..., &requests[0]);
MPI_Irecv(bottomHalo, ..., rank+1, ..., &requests[1]);

// Compute inner region while waiting
computeInnerRegion(...);

// Wait for halo data
MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

// Compute border pixels
computeBorderPixels(...);
```

---

### ⚠️ Requirement 4: MPI_Gatherv for irregular sizes
**Status: ⚠️ PARTIALLY MET**

**Current Implementation:**
- Handles image height not divisible by Nprocs correctly (lines 17-30 in `mpi_utils.cpp`)
- Uses `MPI_Allreduce` to calculate total height
- BUT uses `MPI_Send`/`MPI_Recv` loops instead of `MPI_Gatherv`

**What's Needed:**
- Replace gather loop with `MPI_Gatherv` for efficiency

---

### ✅ Requirement 5: I/O on rank 0
**Status: ✅ IMPLEMENTED**

**Current Implementation:**
- File loading/saving appears to be handled on rank 0 only
- Menu interaction is on rank 0 only
- This requirement is met

---

## Code Locations

### Files to Modify:
1. **`src/mpi_utils.cpp`**:
   - Replace `distributeImage()` to use `MPI_Scatterv`
   - Replace `gatherImage()` to use `MPI_Gatherv`
   - Complete `exchangeBoundaries()` with non-blocking communication
   - Add halo extension functionality

2. **`src/main.cpp`**:
   - Add halo exchange before convolution filter operations
   - Ensure filters use extended images with ghost cells

3. **Filter/Edge Detection Operations**:
   - Need MPI-aware versions that:
     - Exchange halos before processing
     - Process inner region while waiting
     - Process borders after halo arrives

---

## Missing Features Summary

| Requirement | Status | Priority |
|------------|--------|----------|
| MPI_Scatterv/MPI_Gatherv | ❌ Missing | High |
| Halo/Ghost rows | ❌ Incomplete | High |
| Non-blocking communication | ❌ Missing | High |
| Overlap comm & compute | ❌ Missing | High |
| MPI_Gatherv for irregular sizes | ⚠️ Partial | Medium |
| I/O on rank 0 | ✅ Done | - |

---

## Recommendations

1. **High Priority**: Implement `MPI_Scatterv`/`MPI_Gatherv` for efficient data distribution
2. **High Priority**: Complete halo exchange with non-blocking communication
3. **High Priority**: Add overlap of communication and computation
4. **Medium Priority**: Refactor gather to use `MPI_Gatherv` instead of loops

The current implementation works for point operations but does NOT properly support convolution filters that require neighbor pixels.

