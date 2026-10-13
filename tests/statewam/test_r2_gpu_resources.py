import ctypes

import pytest

from scripts.statewam.r2 import gpu_resources as resources


@pytest.mark.parametrize("value", [None, 2**32 - 1, 2**64 - 1,
                                  ctypes.c_uint(2**32 - 1), ctypes.c_ulonglong(2**64 - 1)])
def test_unavailable_process_memory(value):
    assert resources.process_memory_bytes(value) is None


@pytest.mark.parametrize("value", [0, 1024, ctypes.c_ulonglong(1024)])
def test_available_process_memory(value):
    assert resources.process_memory_bytes(value) == getattr(value, "value", value)


def idle(uuid):
    return {"uuid": uuid, "errors": [], "processes": [], "memory_used_bytes": 0,
            "utilization_percent": 0}


def test_project_wide_limit_and_overlap(monkeypatch):
    monkeypatch.setattr(resources, "alive", lambda record: bool(record))
    devices = [idle(str(i)) for i in range(8)]
    leases = {"other": {"supervisor": {"pid": 1}, "gpu_uuids": ["0", "1", "2"], "processes": []}}
    resources.validate_lease(["3"], leases, devices)
    with pytest.raises(ValueError, match="four simultaneously"):
        resources.validate_lease(["3", "4"], leases, devices)
    with pytest.raises(ValueError, match="already leased"):
        resources.validate_lease(["0"], leases, devices)


def test_reject_external_process_and_unobserved_gpu():
    for changes in ({"processes": [{"pid": 999}]},
                    {"errors": [{"operation": "compute"}]},
                    {"memory_used_bytes": 300 * 1024**2}, {"utilization_percent": 90}):
        with pytest.raises(ValueError, match="not demonstrably idle"):
            resources.validate_lease(["0"], {}, [dict(idle("0"), **changes)])


def test_gpu_accounting_counts_devices_not_processes():
    samples = [{"time": t, "rss_bytes": 1, "sampling_seconds": 0.01,
                "used_uuids": ["0", "0"], "devices": [dict(idle("0"), power_watts=100)]}
               for t in (0, 10)]
    report = resources.performance_summary(samples, gpu_uuids=["0"], started=0, ended=10)
    assert report["allocated_gpu_hours"] == pytest.approx(10 / 3600)
    assert report["observed_process_gpu_hours"] == pytest.approx(10 / 3600)
    assert report["estimated_device_kwh"] == pytest.approx(1000 / 3600000)


def test_missing_attribution_is_not_zero_usage():
    samples = [{"time": t, "rss_bytes": 1, "sampling_seconds": 0.01,
                "used_uuids": [], "devices": [dict(idle("0"), power_watts=100)],
                "process_attribution_complete": False} for t in (0, 10)]
    report = resources.performance_summary(samples, gpu_uuids=["0"], started=0, ended=10)
    assert report["observed_process_gpu_hours"] is None
    assert report["estimated_device_kwh"] is None
    assert report["unattributed_sample_count"] == 2


def test_no_samples_do_not_prove_zero_usage():
    report = resources.performance_summary([], gpu_uuids=["0"], started=0, ended=10)
    assert report["observed_process_gpu_hours"] is None
    assert report["estimated_device_kwh"] is None


def test_graphics_spill_detected_with_unmapped_host_pid():
    devices = [dict(idle("4"), processes=[{"pid": 99999, "kinds": ["compute", "graphics"]}]),
               dict(idle("0"), processes=[{"pid": 99999, "kinds": ["graphics"]}]),
               dict(idle("1"), processes=[{"pid": 11111, "kinds": ["compute"]}])]
    assert resources.lease_process_spill(devices, ["4"]) == [{"uuid": "0", "pid": 99999, "kinds": ["graphics"]}]
    assert resources.lease_process_spill(devices, ["0", "4"]) == []
    assert resources.lease_process_spill(devices, ["1"]) == []
