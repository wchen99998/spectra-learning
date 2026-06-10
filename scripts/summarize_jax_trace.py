import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


TPU_V6E_BF16_PEAK_FLOPS = 918e12
CONTAINER_HLO_CATEGORIES = {"while", "conditional"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a JAX profiler trace.")
    parser.add_argument("trace_json", type=Path)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--step-name", default="accumulated_train_step")
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--peak-flops-per-device", type=float, default=TPU_V6E_BF16_PEAK_FLOPS)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def _open_trace(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def _as_int(value: object) -> int:
    return int(value) if value is not None else 0


def _source_key(source_stack: str) -> str:
    first = source_stack.splitlines()[0] if source_stack else "<unknown>"
    cwd = str(Path.cwd())
    return first.replace(cwd + "/", "")


def _process_names(events: list[dict]) -> dict[int, str]:
    names = {}
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            names[int(event["pid"])] = event.get("args", {}).get("name", "")
    return names


def _in_windows(event: dict, windows: list[tuple[float, float]]) -> bool:
    timestamp = float(event.get("ts", 0.0))
    duration = float(event.get("dur", 0.0))
    event_end = timestamp + duration
    for window_start, window_end in windows:
        if timestamp >= window_start and event_end <= window_end:
            return True
    return False


def _summarize_counter(counter: Counter, total: float, top: int) -> list[dict[str, object]]:
    rows = []
    for key, value in counter.most_common(top):
        rows.append(
            {
                "name": key,
                "value": value,
                "percent": 100.0 * float(value) / total if total else 0.0,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    with _open_trace(args.trace_json) as handle:
        trace = json.load(handle)
    events = trace["traceEvents"]
    process_names = _process_names(events)

    flops_by_process: Counter[str] = Counter()
    flops_by_category: Counter[str] = Counter()
    flops_by_source: Counter[str] = Counter()
    flops_by_tf_op: Counter[str] = Counter()
    leaf_flops_by_process: Counter[str] = Counter()
    leaf_flops_by_category: Counter[str] = Counter()
    leaf_flops_by_source: Counter[str] = Counter()
    leaf_flops_by_tf_op: Counter[str] = Counter()
    bytes_by_category: Counter[str] = Counter()
    duration_by_category: Counter[str] = Counter()
    bytes_by_source: Counter[str] = Counter()
    duration_by_source: Counter[str] = Counter()
    data_formatting_duration_by_source: Counter[str] = Counter()
    data_formatting_bytes_by_source: Counter[str] = Counter()
    step_events_by_process: defaultdict[str, list[dict]] = defaultdict(list)

    for event in events:
        if event.get("ph") != "X":
            continue
        process = process_names.get(int(event.get("pid", -1)), str(event.get("pid")))
        name = event.get("name", "")
        if args.step_name in name and process.startswith("/device:TPU"):
            step_events_by_process[process].append(event)

    step_windows_by_process = {
        process: [
            (float(event.get("ts", 0.0)), float(event.get("ts", 0.0)) + float(event.get("dur", 0.0)))
            for event in step_events
        ]
        for process, step_events in step_events_by_process.items()
    }
    step_flops_by_process: Counter[str] = Counter()
    step_leaf_flops_by_process: Counter[str] = Counter()

    for event in events:
        if event.get("ph") != "X":
            continue
        process = process_names.get(int(event.get("pid", -1)), str(event.get("pid")))
        event_args = event.get("args", {})
        if "model_flops" not in event_args:
            continue
        flops = _as_int(event_args.get("model_flops"))
        category = event_args.get("hlo_category", "<unknown>")
        flops_by_process[process] += flops
        flops_by_category[category] += flops
        flops_by_source[_source_key(event_args.get("source_stack", ""))] += flops
        flops_by_tf_op[event_args.get("tf_op", "<unknown>")] += flops
        if category not in CONTAINER_HLO_CATEGORIES:
            leaf_flops_by_process[process] += flops
            leaf_flops_by_category[category] += flops
            leaf_flops_by_source[_source_key(event_args.get("source_stack", ""))] += flops
            leaf_flops_by_tf_op[event_args.get("tf_op", "<unknown>")] += flops
        source = _source_key(event_args.get("source_stack", ""))
        raw_bytes = _as_int(event_args.get("raw_bytes_accessed"))
        duration_ns = int(float(event.get("dur", 0.0)) * 1000.0)
        bytes_by_category[category] += raw_bytes
        duration_by_category[category] += duration_ns
        bytes_by_source[source] += raw_bytes
        duration_by_source[source] += duration_ns
        if category == "data formatting":
            data_formatting_duration_by_source[source] += duration_ns
            data_formatting_bytes_by_source[source] += raw_bytes
        if _in_windows(event, step_windows_by_process.get(process, [])):
            step_flops_by_process[process] += flops
            if category not in CONTAINER_HLO_CATEGORIES:
                step_leaf_flops_by_process[process] += flops

    total_flops = sum(flops_by_process.values())
    leaf_flops = sum(leaf_flops_by_process.values())
    step_window_flops = sum(step_flops_by_process.values())
    step_window_leaf_flops = sum(step_leaf_flops_by_process.values())
    step_counts = [len(value) for value in step_events_by_process.values()]
    inferred_steps = min(step_counts) if step_counts else 0
    flops_per_step = total_flops / inferred_steps if inferred_steps else 0.0
    leaf_flops_per_step = leaf_flops / inferred_steps if inferred_steps else 0.0
    step_window_flops_per_step = (
        step_window_flops / inferred_steps if inferred_steps else 0.0
    )
    step_window_leaf_flops_per_step = (
        step_window_leaf_flops / inferred_steps if inferred_steps else 0.0
    )

    metrics = {}
    if args.metrics_json is not None:
        metrics = json.loads(args.metrics_json.read_text())
    measured_steps_per_second = metrics.get("run/measured_steps_per_second")
    mfu_percent = None
    if measured_steps_per_second is not None and flops_per_step:
        peak = args.devices * args.peak_flops_per_device
        mfu_percent = 100.0 * flops_per_step * float(measured_steps_per_second) / peak
    leaf_mfu_percent = None
    if measured_steps_per_second is not None and leaf_flops_per_step:
        peak = args.devices * args.peak_flops_per_device
        leaf_mfu_percent = (
            100.0 * leaf_flops_per_step * float(measured_steps_per_second) / peak
        )
    step_window_leaf_mfu_percent = None
    if measured_steps_per_second is not None and step_window_leaf_flops_per_step:
        peak = args.devices * args.peak_flops_per_device
        step_window_leaf_mfu_percent = (
            100.0
            * step_window_leaf_flops_per_step
            * float(measured_steps_per_second)
            / peak
        )

    summary = {
        "trace": str(args.trace_json),
        "total_model_flops": total_flops,
        "inferred_profiled_steps": inferred_steps,
        "step_events_by_process": {
            key: len(value) for key, value in sorted(step_events_by_process.items())
        },
        "model_flops_per_step": flops_per_step,
        "leaf_model_flops": leaf_flops,
        "leaf_model_flops_per_step": leaf_flops_per_step,
        "step_window_model_flops": step_window_flops,
        "step_window_model_flops_per_step": step_window_flops_per_step,
        "step_window_leaf_model_flops": step_window_leaf_flops,
        "step_window_leaf_model_flops_per_step": step_window_leaf_flops_per_step,
        "metrics_steps_per_second": measured_steps_per_second,
        "metrics_samples_per_second": metrics.get("run/measured_samples_per_second"),
        "mfu_percent_from_metrics": mfu_percent,
        "leaf_mfu_percent_from_metrics": leaf_mfu_percent,
        "step_window_leaf_mfu_percent_from_metrics": step_window_leaf_mfu_percent,
        "flops_by_process": dict(flops_by_process),
        "leaf_flops_by_process": dict(leaf_flops_by_process),
        "step_window_flops_by_process": dict(step_flops_by_process),
        "step_window_leaf_flops_by_process": dict(step_leaf_flops_by_process),
        "top_flops_by_category": _summarize_counter(
            flops_by_category,
            total_flops,
            args.top,
        ),
        "top_flops_by_source": _summarize_counter(flops_by_source, total_flops, args.top),
        "top_flops_by_tf_op": _summarize_counter(flops_by_tf_op, total_flops, args.top),
        "top_leaf_flops_by_category": _summarize_counter(
            leaf_flops_by_category,
            leaf_flops,
            args.top,
        ),
        "top_leaf_flops_by_source": _summarize_counter(
            leaf_flops_by_source,
            leaf_flops,
            args.top,
        ),
        "top_leaf_flops_by_tf_op": _summarize_counter(
            leaf_flops_by_tf_op,
            leaf_flops,
            args.top,
        ),
        "top_bytes_by_category": _summarize_counter(
            bytes_by_category,
            sum(bytes_by_category.values()),
            args.top,
        ),
        "top_bytes_by_source": _summarize_counter(
            bytes_by_source,
            sum(bytes_by_source.values()),
            args.top,
        ),
        "top_duration_us_by_category": _summarize_counter(
            duration_by_category,
            sum(duration_by_category.values()),
            args.top,
        ),
        "top_duration_us_by_source": _summarize_counter(
            duration_by_source,
            sum(duration_by_source.values()),
            args.top,
        ),
        "top_data_formatting_duration_us_by_source": _summarize_counter(
            data_formatting_duration_by_source,
            sum(data_formatting_duration_by_source.values()),
            args.top,
        ),
        "top_data_formatting_bytes_by_source": _summarize_counter(
            data_formatting_bytes_by_source,
            sum(data_formatting_bytes_by_source.values()),
            args.top,
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
