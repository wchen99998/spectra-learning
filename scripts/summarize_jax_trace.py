import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path


TPU_V6E_BF16_PEAK_FLOPS = 918e12
CONTAINER_HLO_CATEGORIES = {"while", "conditional"}
SOURCE_ALIASES = {
    "spectra_learning/models/pairformer.py": "spectra_learning/models/pairmixer.py",
    "spectra_learning/models/pairformer_jax.py": "spectra_learning/models/pairmixer_jax.py",
}
WRAPPER_SOURCE_FILES = {"spectra_learning/models/common_jax.py"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a JAX profiler trace.")
    parser.add_argument("trace_json", type=Path)
    parser.add_argument("--metrics-json", type=Path, default=None)
    parser.add_argument("--step-name", default="accumulated_train_step")
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--peak-flops-per-device", type=float, default=TPU_V6E_BF16_PEAK_FLOPS)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--stack-depth", type=int, default=4)
    return parser.parse_args()


def _open_trace(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def _as_int(value: object) -> int:
    return int(value) if value is not None else 0


def _trim_source_line(source_line: str) -> str:
    cwd = str(Path.cwd())
    source_line = source_line.replace(cwd + "/", "")
    for old, new in SOURCE_ALIASES.items():
        source_line = source_line.replace(old, new)
    return source_line


def _source_key(source_stack: str) -> str:
    first = source_stack.splitlines()[0] if source_stack else "<unknown>"
    return _trim_source_line(first)


def _stack_key(source_stack: str, depth: int) -> str:
    lines = source_stack.splitlines()[:depth]
    if not lines:
        return "<unknown>"
    return " | ".join(_trim_source_line(line) for line in lines)


def _source_file(source_line: str) -> str:
    source_line = _trim_source_line(source_line)
    return source_line.rsplit(":", 2)[0]


def _op_site_key(source_stack: str) -> str:
    lines = source_stack.splitlines()
    for line in lines:
        source_line = _trim_source_line(line)
        if _source_file(source_line) not in WRAPPER_SOURCE_FILES:
            return source_line
    return _source_key(source_stack)


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
    flops_by_stack: Counter[str] = Counter()
    flops_by_op_site: Counter[str] = Counter()
    flops_by_tf_op: Counter[str] = Counter()
    leaf_flops_by_process: Counter[str] = Counter()
    leaf_flops_by_category: Counter[str] = Counter()
    leaf_flops_by_source: Counter[str] = Counter()
    leaf_flops_by_stack: Counter[str] = Counter()
    leaf_flops_by_op_site: Counter[str] = Counter()
    leaf_flops_by_tf_op: Counter[str] = Counter()
    bytes_by_category: Counter[str] = Counter()
    duration_by_category: Counter[str] = Counter()
    bytes_by_source: Counter[str] = Counter()
    bytes_by_stack: Counter[str] = Counter()
    bytes_by_op_site: Counter[str] = Counter()
    duration_by_source: Counter[str] = Counter()
    duration_by_stack: Counter[str] = Counter()
    duration_by_op_site: Counter[str] = Counter()
    leaf_bytes_by_category: Counter[str] = Counter()
    leaf_duration_by_category: Counter[str] = Counter()
    leaf_bytes_by_source: Counter[str] = Counter()
    leaf_bytes_by_stack: Counter[str] = Counter()
    leaf_bytes_by_op_site: Counter[str] = Counter()
    leaf_duration_by_source: Counter[str] = Counter()
    leaf_duration_by_stack: Counter[str] = Counter()
    leaf_duration_by_op_site: Counter[str] = Counter()
    data_formatting_duration_by_source: Counter[str] = Counter()
    data_formatting_duration_by_stack: Counter[str] = Counter()
    data_formatting_duration_by_op_site: Counter[str] = Counter()
    data_formatting_bytes_by_source: Counter[str] = Counter()
    data_formatting_bytes_by_stack: Counter[str] = Counter()
    data_formatting_bytes_by_op_site: Counter[str] = Counter()
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
    step_leaf_duration_by_category: Counter[str] = Counter()
    step_leaf_duration_by_source: Counter[str] = Counter()
    step_leaf_duration_by_stack: Counter[str] = Counter()
    step_leaf_duration_by_op_site: Counter[str] = Counter()

    for event in events:
        if event.get("ph") != "X":
            continue
        process = process_names.get(int(event.get("pid", -1)), str(event.get("pid")))
        event_args = event.get("args", {})
        if "model_flops" not in event_args:
            continue
        flops = _as_int(event_args.get("model_flops"))
        category = event_args.get("hlo_category", "<unknown>")
        source_stack = event_args.get("source_stack", "")
        source = _source_key(source_stack)
        stack = _stack_key(source_stack, args.stack_depth)
        op_site = _op_site_key(source_stack)
        flops_by_process[process] += flops
        flops_by_category[category] += flops
        flops_by_source[source] += flops
        flops_by_stack[stack] += flops
        flops_by_op_site[op_site] += flops
        flops_by_tf_op[event_args.get("tf_op", "<unknown>")] += flops
        if category not in CONTAINER_HLO_CATEGORIES:
            leaf_flops_by_process[process] += flops
            leaf_flops_by_category[category] += flops
            leaf_flops_by_source[source] += flops
            leaf_flops_by_stack[stack] += flops
            leaf_flops_by_op_site[op_site] += flops
            leaf_flops_by_tf_op[event_args.get("tf_op", "<unknown>")] += flops
        raw_bytes = _as_int(event_args.get("raw_bytes_accessed"))
        duration_us = float(event.get("dur", 0.0))
        bytes_by_category[category] += raw_bytes
        duration_by_category[category] += duration_us
        bytes_by_source[source] += raw_bytes
        bytes_by_stack[stack] += raw_bytes
        bytes_by_op_site[op_site] += raw_bytes
        duration_by_source[source] += duration_us
        duration_by_stack[stack] += duration_us
        duration_by_op_site[op_site] += duration_us
        is_leaf = category not in CONTAINER_HLO_CATEGORIES
        if is_leaf:
            leaf_bytes_by_category[category] += raw_bytes
            leaf_duration_by_category[category] += duration_us
            leaf_bytes_by_source[source] += raw_bytes
            leaf_bytes_by_stack[stack] += raw_bytes
            leaf_bytes_by_op_site[op_site] += raw_bytes
            leaf_duration_by_source[source] += duration_us
            leaf_duration_by_stack[stack] += duration_us
            leaf_duration_by_op_site[op_site] += duration_us
        if category == "data formatting":
            data_formatting_duration_by_source[source] += duration_us
            data_formatting_bytes_by_source[source] += raw_bytes
            data_formatting_duration_by_stack[stack] += duration_us
            data_formatting_bytes_by_stack[stack] += raw_bytes
            data_formatting_duration_by_op_site[op_site] += duration_us
            data_formatting_bytes_by_op_site[op_site] += raw_bytes
        if _in_windows(event, step_windows_by_process.get(process, [])):
            step_flops_by_process[process] += flops
            if is_leaf:
                step_leaf_flops_by_process[process] += flops
                step_leaf_duration_by_category[category] += duration_us
                step_leaf_duration_by_source[source] += duration_us
                step_leaf_duration_by_stack[stack] += duration_us
                step_leaf_duration_by_op_site[op_site] += duration_us

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
    duration_us = sum(duration_by_category.values())
    leaf_duration_us = sum(leaf_duration_by_category.values())
    step_window_leaf_duration_us = sum(step_leaf_duration_by_category.values())
    data_formatting_duration_us = sum(data_formatting_duration_by_source.values())

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
        "duration_us": duration_us,
        "leaf_duration_us": leaf_duration_us,
        "leaf_duration_us_per_step": leaf_duration_us / inferred_steps if inferred_steps else 0.0,
        "step_window_leaf_duration_us": step_window_leaf_duration_us,
        "step_window_leaf_duration_us_per_step": (
            step_window_leaf_duration_us / inferred_steps if inferred_steps else 0.0
        ),
        "data_formatting_duration_us": data_formatting_duration_us,
        "data_formatting_duration_us_per_step": (
            data_formatting_duration_us / inferred_steps if inferred_steps else 0.0
        ),
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
        "top_flops_by_stack": _summarize_counter(flops_by_stack, total_flops, args.top),
        "top_flops_by_op_site": _summarize_counter(
            flops_by_op_site,
            total_flops,
            args.top,
        ),
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
        "top_leaf_flops_by_stack": _summarize_counter(
            leaf_flops_by_stack,
            leaf_flops,
            args.top,
        ),
        "top_leaf_flops_by_op_site": _summarize_counter(
            leaf_flops_by_op_site,
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
        "top_bytes_by_stack": _summarize_counter(
            bytes_by_stack,
            sum(bytes_by_stack.values()),
            args.top,
        ),
        "top_bytes_by_op_site": _summarize_counter(
            bytes_by_op_site,
            sum(bytes_by_op_site.values()),
            args.top,
        ),
        "top_leaf_bytes_by_category": _summarize_counter(
            leaf_bytes_by_category,
            sum(leaf_bytes_by_category.values()),
            args.top,
        ),
        "top_leaf_bytes_by_source": _summarize_counter(
            leaf_bytes_by_source,
            sum(leaf_bytes_by_source.values()),
            args.top,
        ),
        "top_leaf_bytes_by_stack": _summarize_counter(
            leaf_bytes_by_stack,
            sum(leaf_bytes_by_stack.values()),
            args.top,
        ),
        "top_leaf_bytes_by_op_site": _summarize_counter(
            leaf_bytes_by_op_site,
            sum(leaf_bytes_by_op_site.values()),
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
        "top_duration_us_by_stack": _summarize_counter(
            duration_by_stack,
            sum(duration_by_stack.values()),
            args.top,
        ),
        "top_duration_us_by_op_site": _summarize_counter(
            duration_by_op_site,
            sum(duration_by_op_site.values()),
            args.top,
        ),
        "top_leaf_duration_us_by_category": _summarize_counter(
            leaf_duration_by_category,
            sum(leaf_duration_by_category.values()),
            args.top,
        ),
        "top_leaf_duration_us_by_source": _summarize_counter(
            leaf_duration_by_source,
            sum(leaf_duration_by_source.values()),
            args.top,
        ),
        "top_leaf_duration_us_by_stack": _summarize_counter(
            leaf_duration_by_stack,
            sum(leaf_duration_by_stack.values()),
            args.top,
        ),
        "top_leaf_duration_us_by_op_site": _summarize_counter(
            leaf_duration_by_op_site,
            sum(leaf_duration_by_op_site.values()),
            args.top,
        ),
        "top_step_window_leaf_duration_us_by_category": _summarize_counter(
            step_leaf_duration_by_category,
            sum(step_leaf_duration_by_category.values()),
            args.top,
        ),
        "top_step_window_leaf_duration_us_by_source": _summarize_counter(
            step_leaf_duration_by_source,
            sum(step_leaf_duration_by_source.values()),
            args.top,
        ),
        "top_step_window_leaf_duration_us_by_stack": _summarize_counter(
            step_leaf_duration_by_stack,
            sum(step_leaf_duration_by_stack.values()),
            args.top,
        ),
        "top_step_window_leaf_duration_us_by_op_site": _summarize_counter(
            step_leaf_duration_by_op_site,
            sum(step_leaf_duration_by_op_site.values()),
            args.top,
        ),
        "top_data_formatting_duration_us_by_source": _summarize_counter(
            data_formatting_duration_by_source,
            sum(data_formatting_duration_by_source.values()),
            args.top,
        ),
        "top_data_formatting_duration_us_by_stack": _summarize_counter(
            data_formatting_duration_by_stack,
            sum(data_formatting_duration_by_stack.values()),
            args.top,
        ),
        "top_data_formatting_duration_us_by_op_site": _summarize_counter(
            data_formatting_duration_by_op_site,
            sum(data_formatting_duration_by_op_site.values()),
            args.top,
        ),
        "top_data_formatting_bytes_by_source": _summarize_counter(
            data_formatting_bytes_by_source,
            sum(data_formatting_bytes_by_source.values()),
            args.top,
        ),
        "top_data_formatting_bytes_by_stack": _summarize_counter(
            data_formatting_bytes_by_stack,
            sum(data_formatting_bytes_by_stack.values()),
            args.top,
        ),
        "top_data_formatting_bytes_by_op_site": _summarize_counter(
            data_formatting_bytes_by_op_site,
            sum(data_formatting_bytes_by_op_site.values()),
            args.top,
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
