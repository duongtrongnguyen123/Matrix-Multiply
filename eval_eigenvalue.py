#!/usr/bin/env python3
"""
Compute smallest eigenvalue of matrices for different compute dtypes.
Uses shift-invert mode for efficient computation of smallest eigenvalue.
"""

import numpy as np
import struct
import sys
from pathlib import Path
from scipy.sparse.linalg import eigsh


def load_matrix_bin(filepath: str) -> np.ndarray:
    """Load binary matrix file with int64 header + float32 data."""
    with open(filepath, 'rb') as f:
        N = struct.unpack('<q', f.read(8))[0]
        data = np.frombuffer(f.read(N * N * 4), dtype=np.float32)
        return data.reshape(N, N)


def compute_smallest_eigenvalue(matrix: np.ndarray, k: int = 1) -> float:
    """
    Compute smallest eigenvalue using ARPACK's shift-invert mode.
    Much faster than full eigendecomposition for large matrices.
    """
    # Use shift-invert with sigma=0 to find smallest eigenvalues
    # which='LM' with sigma finds eigenvalues closest to sigma
    eigenvalues, _ = eigsh(matrix.astype(np.float64), k=k, which='SM', tol=1e-8)
    return np.min(eigenvalues)


def main():
    output_dir = Path("output_dir")
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])

    dtypes = ['fp32', 'bf16', 'tf32']
    results = {}

    print("Computing smallest eigenvalue for each dtype...")
    print("=" * 60)
    sys.stdout.flush()

    for dtype in dtypes:
        bin_file = list(output_dir.glob(f"C_*_{dtype}.bin"))
        if not bin_file:
            print(f"  {dtype}: file not found")
            continue

        bin_file = bin_file[0]
        print(f"\n[{dtype}] Loading {bin_file.name}...", flush=True)

        matrix = load_matrix_bin(bin_file)
        print(f"  Shape: {matrix.shape}", flush=True)

        print(f"  Computing smallest eigenvalue (ARPACK)...", flush=True)
        try:
            eig_min = compute_smallest_eigenvalue(matrix)
            results[dtype] = eig_min
            print(f"  Smallest eigenvalue: {eig_min:.10e}", flush=True)
        except Exception as e:
            print(f"  Error: {e}", flush=True)

    # Summary table
    print("\n" + "=" * 60)
    print("  Summary: Smallest Eigenvalues")
    print("=" * 60)
    print(f"{'dtype':<10} {'smallest eigenvalue':>25}")
    print("-" * 60)

    for dtype in dtypes:
        if dtype in results:
            print(f"{dtype:<10} {results[dtype]:>25.10e}")

    # Compare vs fp32
    if 'fp32' in results:
        print("\n" + "-" * 60)
        print("Difference from fp32 reference:")
        ref = results['fp32']
        for dtype in ['bf16', 'tf32']:
            if dtype in results:
                diff = results[dtype] - ref
                rel_diff = abs(diff / ref) * 100 if ref != 0 else float('nan')
                print(f"  {dtype}: {diff:+.10e} ({rel_diff:.6f}% relative)")

    print("=" * 60)


if __name__ == "__main__":
    main()
