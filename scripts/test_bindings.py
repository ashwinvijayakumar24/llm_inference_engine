"""Task 0.5.2/0.5.3 — verify CPU and CUDA nanobind round-trips."""
import sys
import numpy as np

sys.path.insert(0, "build")
import engine_kernels

def test_cpu():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = engine_kernels.add_one(arr)
    expected = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    assert np.allclose(result, expected), f"CPU failed: {result}"
    print("PASS add_one (CPU)")

def test_cuda():
    arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    result = engine_kernels.add_one_cuda(arr)
    expected = np.array([11.0, 21.0, 31.0], dtype=np.float32)
    assert np.allclose(result, expected), f"CUDA failed: {result}"
    print("PASS add_one_cuda (CUDA)")

if __name__ == "__main__":
    test_cpu()
    test_cuda()
    print("All binding tests passed.")
