#!/usr/bin/env python3
"""
Compute smallest and largest eigenvalues of matrices for different compute dtypes.
Uses ARPACK for efficient computation. Reference: fp64.
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


def compute_eigenvalues(matrix: np.ndarray) -> tuple:
    """Compute smallest and largest eigenvalue using ARPACK."""
    mat64 = matrix.astype(np.float64)
    eig_sm, _ = eigsh(mat64, k=1, which='SM', tol=1e-10)
    eig_lm, _ = eigsh(mat64, k=1, which='LM', tol=1e-10)
    return eig_sm[0], eig_lm[0]


def main():
    output_dir = Path("output_dir")
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f"Error: Directory '{output_dir}' not found")
        print(f"Usage: {sys.argv[0]} [output_dir]")
        sys.exit(1)

    # All dtypes, fp64 first as reference
    dtypes = ['fp64', 'fp32', 'tf32', 'bf16', 'fp16']
    results = {}

    print("Computing eigenvalues for each dtype (reference: fp64)")
    print("=" * 70)
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
        print(f"  Computing eigenvalues (ARPACK)...", flush=True)

        try:
            eig_sm, eig_lm = compute_eigenvalues(matrix)
            results[dtype] = {'smallest': eig_sm, 'largest': eig_lm}
            print(f"  Smallest: {eig_sm:.10e}", flush=True)
            print(f"  Largest:  {eig_lm:.10e}", flush=True)
        except Exception as e:
            print(f"  Error: {e}", flush=True)

    # Summary table
    print("\n" + "=" * 90)
    print("  Summary: Eigenvalues (reference: fp64)")
    print("=" * 90)
    print(f"{'dtype':<8} {'smallest':>22} {'diff':>15} {'largest':>22} {'diff':>15}")
    print("-" * 90)

    ref_sm = results.get('fp64', {}).get('smallest', 0)
    ref_lm = results.get('fp64', {}).get('largest', 0)

    for dtype in dtypes:
        if dtype in results:
            sm = results[dtype]['smallest']
            lm = results[dtype]['largest']
            diff_sm = sm - ref_sm if ref_sm else 0
            diff_lm = lm - ref_lm if ref_lm else 0
            print(f"{dtype:<8} {sm:>22.10e} {diff_sm:>+15.4e} {lm:>22.10e} {diff_lm:>+15.4e}")

    # Relative errors
    if 'fp64' in results:
        print("\n" + "-" * 60)
        print("Relative errors vs fp64:")
        print(f"{'dtype':<8} {'smallest rel%':>18} {'largest rel%':>18}")
        print("-" * 60)
        for dtype in dtypes:
            if dtype != 'fp64' and dtype in results:
                sm = results[dtype]['smallest']
                lm = results[dtype]['largest']
                rel_sm = abs(sm - ref_sm) / ref_sm * 100 if ref_sm else 0
                rel_lm = abs(lm - ref_lm) / ref_lm * 100 if ref_lm else 0
                print(f"{dtype:<8} {rel_sm:>17.6f}% {rel_lm:>17.6f}%")

    print("=" * 90)


if __name__ == "__main__":
    main()
