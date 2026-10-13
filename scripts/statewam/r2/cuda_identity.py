"""Verify actual CUDA logical devices against the job's leased GPU UUIDs."""

import ctypes
import os
import uuid


def verify_cuda_devices():
    expected = os.environ["R2_GPU_UUIDS"].split(",")
    driver = ctypes.CDLL("libcuda.so.1")

    def checked(name, *args):
        result = getattr(driver, name)(*args)
        if result:
            raise RuntimeError(f"{name} failed with CUDA status {result}")

    checked("cuInit", 0)
    count = ctypes.c_int()
    checked("cuDeviceGetCount", ctypes.byref(count))
    actual = []
    for index in range(count.value):
        device = ctypes.c_int()
        checked("cuDeviceGet", ctypes.byref(device), index)
        value = (ctypes.c_ubyte * 16)()
        checked("cuDeviceGetUuid", ctypes.byref(value), device)
        actual.append("GPU-" + str(uuid.UUID(bytes=bytes(value))))
    if actual != expected:
        raise RuntimeError(f"CUDA devices do not match lease: expected={expected}, actual={actual}")
    return actual
