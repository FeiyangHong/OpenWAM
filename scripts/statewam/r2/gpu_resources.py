"""NVML samples and conservative, project-wide GPU lease validation."""

import statistics

import pynvml as nvml

from scripts.statewam.r2.cpu_job import alive


def process_memory_bytes(value):
    """Normalize Python/ctypes NVML memory counters and unavailable sentinels."""
    value = getattr(value, "value", value)
    if value in (None, 2**32 - 1, 2**64 - 1):
        return None
    return int(value)


class DeviceMonitor:
    def __init__(self):
        nvml.nvmlInit()
        self.handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(nvml.nvmlDeviceGetCount())]

    def close(self):
        nvml.nvmlShutdown()

    def snapshot(self):
        devices = []
        for index, handle in enumerate(self.handles):
            try:
                memory = nvml.nvmlDeviceGetMemoryInfo(handle, version=nvml.nvmlMemory_v2)
                memory_api = "v2_used_excludes_driver_reserved"
            except (AttributeError, TypeError, nvml.NVMLError):
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                memory_api = "v1_used_includes_driver_reserved"
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            processes, errors = {}, []
            for kind, getter in (("compute", nvml.nvmlDeviceGetComputeRunningProcesses),
                                 ("graphics", nvml.nvmlDeviceGetGraphicsRunningProcesses)):
                try:
                    for process in getter(handle):
                        row = processes.setdefault(process.pid, {"pid": process.pid, "kinds": [], "memory_bytes": None})
                        row["kinds"].append(kind)
                        used_memory = process_memory_bytes(process.usedGpuMemory)
                        if used_memory is not None:
                            row["memory_bytes"] = used_memory
                except nvml.NVMLError as error:
                    errors.append({"operation": kind, "error": type(error).__name__})
            try:
                watts = nvml.nvmlDeviceGetPowerUsage(handle) / 1000
            except nvml.NVMLError as error:
                watts = None
                errors.append({"operation": "power", "error": type(error).__name__})
            devices.append({"index": index, "uuid": nvml.nvmlDeviceGetUUID(handle),
                            "name": nvml.nvmlDeviceGetName(handle), "memory_total_bytes": memory.total,
                            "memory_reserved_bytes": getattr(memory, "reserved", None), "memory_api": memory_api,
                            "memory_used_bytes": memory.used, "utilization_percent": utilization.gpu,
                            "memory_utilization_percent": utilization.memory, "power_watts": watts,
                            "processes": list(processes.values()), "errors": errors})
        return devices


def validate_lease(requested, leases, devices):
    if not requested or len(set(requested)) != len(requested) or len(requested) > 4:
        raise ValueError("Require one to four distinct GPUs")
    busy = set()
    for lease in leases.values():
        if alive(lease.get("supervisor")) or any(alive(p) for p in lease.get("processes", [])):
            busy.update(lease["gpu_uuids"])
    if busy.intersection(requested):
        raise ValueError("Requested GPU is already leased to this Goal")
    if len(busy.union(requested)) > 4:
        raise ValueError("Project-wide limit of four simultaneously leased GPUs")
    available = {row["uuid"]: row for row in devices}
    for uuid in requested:
        if uuid not in available:
            raise ValueError("Unknown GPU UUID")
        row = available[uuid]
        process_errors = [error for error in row["errors"] if error["operation"] in ("compute", "graphics")]
        if process_errors or row["processes"] or row["memory_used_bytes"] > 256 * 1024**2 or row["utilization_percent"] > 5:
            raise ValueError(f"GPU not demonstrably idle: {uuid}")


def lease_process_spill(devices, leased_uuids):
    """Link NVML PIDs across devices without assuming container PID equality.

    A PID observed on an exclusive lease and another device is suspicious even
    when it cannot be mapped into this container. Stop only the owned workload;
    this correlation is not authority to signal the unmatched host PID.
    """
    leased = set(leased_uuids)
    pids = {process["pid"] for row in devices if row["uuid"] in leased for process in row["processes"]}
    return [{"uuid": row["uuid"], **process} for row in devices if row["uuid"] not in leased
            for process in row["processes"] if process["pid"] in pids]


def distribution(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    percentiles = statistics.quantiles(values, n=100, method="inclusive") if len(values) > 1 else [values[0]] * 99
    return {"count": len(values), "mean": statistics.mean(values), "p50": percentiles[49],
            "p95": percentiles[94], "max": max(values)}


def performance_summary(samples, *, gpu_uuids, started, ended):
    gaps = [b["time"] - a["time"] for a, b in zip(samples, samples[1:])]
    result = {"wall_seconds": ended - started, "sample_count": len(samples),
              "allocated_gpu_hours": len(set(gpu_uuids)) * (ended - started) / 3600,
              "observed_process_gpu_hours": 0, "estimated_device_kwh": 0,
              "max_sample_gap_seconds": max(gaps, default=None),
              "peak_process_tree_rss_bytes": max((r["rss_bytes"] for r in samples), default=None),
              "measurement_seconds_total": sum(row["sampling_seconds"] for row in samples), "by_gpu": {}}
    for previous, current in zip(samples, samples[1:]):
        duration = current["time"] - previous["time"]
        result["observed_process_gpu_hours"] += duration * len(set(previous["used_uuids"])) / 3600
        power = [row["power_watts"] for row in previous["devices"] if row["uuid"] in gpu_uuids]
        if result["estimated_device_kwh"] is not None:
            if any(watts is None for watts in power):
                result["estimated_device_kwh"] = None
            else:
                result["estimated_device_kwh"] += duration * sum(power) / 3600000
    for uuid in gpu_uuids:
        rows = [device for sample in samples for device in sample["devices"] if device["uuid"] == uuid]
        result["by_gpu"][uuid] = {key: distribution(row[key] for row in rows) for key in (
            "utilization_percent", "memory_used_bytes", "power_watts")}
    result["unattributed_sample_count"] = sum(not sample.get("process_attribution_complete", True) for sample in samples)
    if result["unattributed_sample_count"]:
        result["observed_process_gpu_hours"] = None
        result["estimated_device_kwh"] = None
    if len(samples) < 2:
        result["observed_process_gpu_hours"] = None
        result["estimated_device_kwh"] = None
    result["accounting_note"] = (
        "Exclusive leases include startup/checkpoint/service waiting. Process GPU-time is left-endpoint sampled, "
        "one count per GPU even with multiple processes. Energy is whole-device sampled power, not incremental model energy. "
        "CPU RSS sums can double-count shared pages. Framework allocated/reserved must be recorded by the workload."
        " Unattributed processes invalidate process GPU-hours and project energy; lease-hours remain allocation cost."
    )
    return result
