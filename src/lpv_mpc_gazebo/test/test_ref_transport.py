"""Phase-0 reference-topic transport contract.

Checks the only non-trivial part of the CSV->topic swap: a flat, row-major
(N, 6) array survives ravel (publisher) -> reshape (consumer), and the
consumer's size guard accepts exactly the valid shapes. Pure numpy, no ROS.

Run: python3 test/test_ref_transport.py   (or via pytest)
"""
import numpy as np


def valid_ref_size(n):
    # mirrors the guard in lpv_mpc_node._ref_path_callback
    return n >= 12 and n % 6 == 0


def test_roundtrip():
    wp = np.arange(5 * 6, dtype=float).reshape(5, 6)   # [s,x,y,psi,kappa,vx]
    flat = wp[:, :6].astype(float).ravel().tolist()    # publisher side
    assert valid_ref_size(len(flat))
    back = np.asarray(flat, dtype=float).reshape(-1, 6)  # consumer side
    assert np.array_equal(back, wp)


def test_size_guard():
    assert valid_ref_size(12) and valid_ref_size(6000)
    assert not valid_ref_size(6)    # single point -> dedup/diff needs >= 2
    assert not valid_ref_size(13)   # not a multiple of 6
    assert not valid_ref_size(0)


if __name__ == '__main__':
    test_roundtrip()
    test_size_guard()
    print('ok')
