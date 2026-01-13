#!/usr/bin/env python3
"""
Evaluate matrix multiplication results across different compute dtypes.
Uses fp32 as reference and computes error metrics for bf16 and tf32.
"""

import numpy as np
import struct
import os
import sys
from pathlib import Path


def load_matrix_bin(filepath: str) -> np.ndarray:
    """Load binary matrix file with int64 header + float32 data."""
    with open(filepath, 'rb') as f:
        N = struct.unpack('<q', f.read(8))[0]  # little-endian int64
        data = np.frombuffer(f.read(N * N * 4), dtype=np.float32)
        return data.reshape(N, N)


def compute_metrics(ref: np.ndarray, test: np.ndarray, name: str) -> dict:
    """Compute error metrics between reference and test matrices."""
    diff = test.astype(np.float64) - ref.astype(np.float64)
    abs_diff = np.abs(diff)

    # Absolute errors
    max_abs_err = np.max(abs_diff)
    mean_abs_err = np.mean(abs_diff)
    rmse = np.sqrt(np.mean(diff ** 2))

    # Relative errors (avoid division by zero)
    ref_abs = np.abs(ref.astype(np.float64))
    nonzero_mask = ref_abs > 1e-10
    if np.any(nonzero_mask):
        rel_err = abs_diff[nonzero_mask] / ref_abs[nonzero_mask]
        max_rel_err = np.max(rel_err)
        mean_rel_err = np.mean(rel_err)
    else:
        max_rel_err = float('nan')
        mean_rel_err = float('nan')

    # Frobenius norm ratio
    ref_norm = np.linalg.norm(ref.astype(np.float64), 'fro')
    diff_norm = np.linalg.norm(diff, 'fro')
    fro_ratio = diff_norm / ref_norm if ref_norm > 0 else float('nan')

    # Element-wise statistics
    num_exact = np.sum(diff == 0)
    total = diff.size

    return {
        'name': name,
        'max_abs_err': max_abs_err,
        'mean_abs_err': mean_abs_err,
        'rmse': rmse,
        'max_rel_err': max_rel_err,
        'mean_rel_err': mean_rel_err,
        'fro_ratio': fro_ratio,
        'exact_matches': num_exact,
        'total_elements': total,
        'exact_pct': 100.0 * num_exact / total
    }


def print_metrics(metrics: dict):
    """Print metrics in a formatted way."""
    print(f"\n{'=' * 50}")
    print(f"  {metrics['name']} vs fp32 (reference)")
    print(f"{'=' * 50}")
    print(f"  Max absolute error:    {metrics['max_abs_err']:.6e}")
    print(f"  Mean absolute error:   {metrics['mean_abs_err']:.6e}")
    print(f"  RMSE:                  {metrics['rmse']:.6e}")
    print(f"  Max relative error:    {metrics['max_rel_err']:.6e}")
    print(f"  Mean relative error:   {metrics['mean_rel_err']:.6e}")
    print(f"  Frobenius norm ratio:  {metrics['fro_ratio']:.6e}")
    print(f"  Exact matches:         {metrics['exact_matches']:,} / {metrics['total_elements']:,} ({metrics['exact_pct']:.2f}%)")


def print_summary_table(all_metrics: list):
    """Print a summary comparison table."""
    dtypes = ['fp32', 'tf32', 'bf16', 'fp16']
    metrics_dict = {m['name'].lower(): m for m in all_metrics}

    print(f"\n{'=' * 95}")
    print("  Summary Comparison Table (vs fp64 reference)")
    print(f"{'=' * 95}")
    print(f"{'Metric':<20} {'fp32':>17} {'tf32':>17} {'bf16':>17} {'fp16':>17}")
    print(f"{'-' * 95}")

    def get_val(dtype, key):
        return metrics_dict.get(dtype, {}).get(key, float('nan'))

    print(f"{'Max abs error':<20}", end='')
    for d in dtypes:
        print(f" {get_val(d, 'max_abs_err'):>16.6e}", end='')
    print()

    print(f"{'Mean abs error':<20}", end='')
    for d in dtypes:
        print(f" {get_val(d, 'mean_abs_err'):>16.6e}", end='')
    print()

    print(f"{'RMSE':<20}", end='')
    for d in dtypes:
        print(f" {get_val(d, 'rmse'):>16.6e}", end='')
    print()

    print(f"{'Max rel error':<20}", end='')
    for d in dtypes:
        print(f" {get_val(d, 'max_rel_err'):>16.6e}", end='')
    print()

    print(f"{'Mean rel error':<20}", end='')
    for d in dtypes:
        print(f" {get_val(d, 'mean_rel_err'):>16.6e}", end='')
    print()

    print(f"{'Frobenius ratio':<20}", end='')
    for d in dtypes:
        print(f" {get_val(d, 'fro_ratio'):>16.6e}", end='')
    print()

    print(f"{'Exact match %':<20}", end='')
    for d in dtypes:
        val = get_val(d, 'exact_pct')
        print(f" {val:>15.2f}%", end='')
    print()

    print(f"{'=' * 95}")


def main():
    output_dir = Path("output_dir")

    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f"Error: Directory '{output_dir}' not found")
        sys.exit(1)

    # Find matrix files
    bin_files = list(output_dir.glob("C_*_*.bin"))
    if not bin_files:
        print(f"Error: No matrix files found in '{output_dir}'")
        sys.exit(1)

    print(f"Found {len(bin_files)} matrix files in '{output_dir}':")
    for f in bin_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")

    # Load reference (fp64)
    fp64_file = output_dir / next((f.name for f in bin_files if 'fp64' in f.name.lower()), None)
    if not fp64_file or not fp64_file.exists():
        print("Error: fp64 reference file not found")
        sys.exit(1)

    print(f"\nLoading reference: {fp64_file.name}...")
    ref_matrix = load_matrix_bin(fp64_file)
    print(f"  Shape: {ref_matrix.shape}")
    print(f"  Range: [{ref_matrix.min():.6e}, {ref_matrix.max():.6e}]")
    print(f"  Mean:  {ref_matrix.mean():.6e}")

    # Evaluate other dtypes
    all_metrics = []
    for bin_file in bin_files:
        if 'fp64' in bin_file.name.lower():
            continue

        # Extract dtype name from filename
        dtype_name = bin_file.stem.split('_')[-1]

        print(f"\nLoading {bin_file.name}...")
        test_matrix = load_matrix_bin(bin_file)
        print(f"  Shape: {test_matrix.shape}")
        print(f"  Range: [{test_matrix.min():.6e}, {test_matrix.max():.6e}]")

        metrics = compute_metrics(ref_matrix, test_matrix, dtype_name)
        all_metrics.append(metrics)
        print_metrics(metrics)

    # Print summary table
    if len(all_metrics) > 1:
        print_summary_table(all_metrics)


if __name__ == "__main__":
    main()
