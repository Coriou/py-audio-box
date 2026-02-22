#!/usr/bin/env python3
"""
tests/remote/lib/report.py
Benchmark report generator for the py-audio-box synthesis test suite.

Reads per-test JSON records from $BENCH_DIR, gathers system + software info,
writes a consolidated benchmark JSON and prints a human-readable RTF table.

Usage (called by run-all.sh):
    python3 /app/tests/remote/lib/report.py \
        --bench-dir  /work/remote-tests/.bench_records \
        --out-dir    /work/remote-tests \
        --target     gpu \
        --run-id     20260222_005313 \
        --wall-s     2583
"""
import argparse
import datetime
import glob
import json
import multiprocessing
import os
import platform
import subprocess
import sys


# ── System info collection ────────────────────────────────────────────────────

def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def gather_hw_info() -> dict:
    hw: dict = {}

    # GPU info via torch
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            hw["gpu"]          = p.name
            hw["gpu_vram_gb"]  = round(p.total_memory / 1024 ** 3, 1)
            hw["gpu_sm_count"] = p.multi_processor_count
            hw["cuda_version"] = torch.version.cuda
            # CU compute capability
            hw["compute_cap"]  = f"{p.major}.{p.minor}"
        else:
            hw["gpu"] = "CPU"
    except ImportError:
        hw["gpu"] = "unknown"

    # nvidia-smi power + clocks
    try:
        smi = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=power.draw,clocks.gr,clocks.mem,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().split(",")
        hw["gpu_power_w"]       = _safe(lambda: float(smi[0]))
        hw["gpu_clock_mhz"]     = _safe(lambda: int(smi[1]))
        hw["gpu_mem_clock_mhz"] = _safe(lambda: int(smi[2]))
        hw["gpu_temp_c"]        = _safe(lambda: int(smi[3]))
        hw["gpu_util_pct"]      = _safe(lambda: int(smi[4]))
    except Exception:
        pass

    # CPU
    hw["cpu_logical_cores"] = multiprocessing.cpu_count()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    hw["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        hw["cpu_model"] = platform.processor() or "unknown"

    # RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    hw["ram_gb"] = round(int(line.split()[1]) / 1024 ** 2, 1)
                    break
    except Exception:
        pass

    return hw


def gather_sw_info(dtype: str) -> dict:
    sw: dict = {"dtype": dtype, "os": platform.system()}

    try:
        import torch  # noqa: PLC0415
        sw["torch"]    = torch.__version__
        sw["cuda"]     = torch.version.cuda or "none"
        sw["python"]   = platform.python_version()
    except ImportError:
        pass

    try:
        import flash_attn  # noqa: PLC0415
        sw["flash_attn"]  = flash_attn.__version__
        sw["attn_impl"]   = "flash_attention_2"
    except Exception as exc:
        sw["flash_attn"] = None
        sw["attn_impl"]  = "sdpa"
        # Surface the root cause (useful for debugging ABI mismatches)
        sw["flash_attn_error"] = str(exc).split("\n")[0][:120]

    try:
        import soundfile as sf  # noqa: PLC0415
        sw["soundfile"] = sf.__version__
    except Exception:
        pass

    return sw


# ── Report rendering ──────────────────────────────────────────────────────────

def render_table(tests: list[dict], hw: dict, sw: dict, summary: dict) -> str:
    lines: list[str] = []

    gpu_label = hw.get("gpu", "?")
    target    = tests[0]["target"] if tests else "?"
    run_id    = summary.get("run_id", "?")

    lines.append("")
    lines.append("╔══════════════════════════════════════════════════════════════════════════════╗")
    lines.append(f"║  BENCHMARK REPORT  [{target}]  {gpu_label:<42}║")
    lines.append(f"║  run_id: {run_id:<68}║")
    lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")
    lines.append(f"║  torch {sw.get('torch','?'):<9}  "
                 f"cuda {sw.get('cuda','?'):<8}  "
                 f"dtype {sw.get('dtype','?'):<12}  "
                 f"flash-attn {sw.get('flash_attn','none')!s:<12}  attn={sw.get('attn_impl','?'):<20}║")
    lines.append(f"║  CPU: {hw.get('cpu_model','?')[:36]:<36}  cores={hw.get('cpu_logical_cores','?'):<4}  "
                 f"RAM={hw.get('ram_gb','?')!s:<6} GB                   ║")
    lines.append("╠══════════════════════════════════════════════════════════════════════════════╣")

    # Column header
    hdr = f"  {'TEST':<52}  {'OK':>3}  {'AUDIO':>6}  {'SYNTH':>6}  {'RTF':>6}  {'WALL':>5}"
    lines.append(f"║{hdr:<78}║")
    lines.append("╠" + "═" * 78 + "╣")

    last_suite = ""
    for t in tests:
        suite = t.get("suite", "")
        if suite != last_suite:
            lines.append(f"║  ── {suite:<73}║")
            last_suite = suite

        label   = t.get("label", "?")[:52]
        ok      = "✓" if t.get("pass") else "✗"
        audio_s = f"{t['audio_s']:.1f}s" if t.get("audio_s") is not None else "  n/a"
        synth_s = f"{t['synth_s']:.1f}s" if t.get("synth_s") is not None else "  n/a"
        rtf     = f"{t['rtf']:.2f}x"     if t.get("rtf")     is not None else "  n/a"
        wall_s  = f"{t['elapsed_s']}s"   if t.get("elapsed_s") is not None else "  n/a"
        row = f"  {label:<52}  {ok:>3}  {audio_s:>6}  {synth_s:>6}  {rtf:>6}  {wall_s:>5}"
        lines.append(f"║{row:<78}║")

    lines.append("╠" + "═" * 78 + "╣")

    # Summary row
    avg_rtf    = summary.get("avg_rtf")
    total_aud  = summary.get("total_audio_s", 0)
    total_syn  = summary.get("total_synth_s", 0)
    wall_total = summary.get("wall_s", 0)
    n_pass     = summary.get("pass", 0)
    n_fail     = summary.get("fail", 0)
    n_total    = summary.get("total", 0)

    lines.append(f"║  {'SUMMARY':<52}  "
                 f"{'pass=' + str(n_pass) + '/' + str(n_total):>9}  "
                 f"{'tot=' + str(round(total_aud)) + 's':>7}  "
                 f"{'syn=' + str(round(total_syn)) + 's':>7}  "
                 f"{'avg=' + (f'{avg_rtf:.2f}x' if avg_rtf else 'n/a'):>7}  "
                 f"{'w=' + str(wall_total) + 's':>5}  ║")
    lines.append("╚" + "═" * 78 + "╝")

    if n_fail:
        lines.append(f"\n  ✗  {n_fail} test(s) FAILED")
        for t in tests:
            if not t.get("pass"):
                lines.append(f"       • {t.get('suite','')}  {t.get('label','')}")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="py-audio-box benchmark report generator")
    p.add_argument("--bench-dir",  required=True,  help="Directory with per-test *.json records")
    p.add_argument("--out-dir",    required=True,  help="Root output directory (benchmarks/ written here)")
    p.add_argument("--target",     default="gpu",  help="Target label (gpu / cpu / rog)")
    p.add_argument("--run-id",     default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.add_argument("--wall-s",     type=int, default=0, help="Total wall-clock seconds for the whole run")
    args = p.parse_args()

    bench_dir = args.bench_dir
    out_dir   = args.out_dir
    target    = args.target
    run_id    = args.run_id

    # ── Collect per-test records ──────────────────────────────────────────────
    records = []
    for fn in sorted(glob.glob(os.path.join(bench_dir, "*.json"))):
        try:
            with open(fn) as f:
                records.append(json.load(f))
        except Exception as exc:
            print(f"  [report] warning: could not read {fn}: {exc}", file=sys.stderr)

    dtype = records[0].get("dtype", args.target) if records else "unknown"

    # ── System info ───────────────────────────────────────────────────────────
    hw = gather_hw_info()
    sw = gather_sw_info(dtype)

    # ── Aggregates ────────────────────────────────────────────────────────────
    pass_cnt  = sum(1 for r in records if r.get("pass"))
    fail_cnt  = sum(1 for r in records if not r.get("pass"))
    rtfs      = [r["rtf"] for r in records if r.get("rtf") is not None]
    avg_rtf   = round(sum(rtfs) / len(rtfs), 3) if rtfs else None
    total_aud = round(sum(r.get("audio_s") or 0 for r in records), 1)
    total_syn = round(sum(r.get("synth_s") or 0 for r in records), 1)

    # GPU name for filename (space → -, strip special chars)
    gpu_slug = hw.get("gpu", "cpu").replace(" ", "-").replace("/", "-")
    gpu_slug = "".join(c for c in gpu_slug if c.isalnum() or c in "-_")

    summary = {
        "run_id":        run_id,
        "pass":          pass_cnt,
        "fail":          fail_cnt,
        "total":         len(records),
        "avg_rtf":       avg_rtf,
        "total_audio_s": total_aud,
        "total_synth_s": total_syn,
        "wall_s":        args.wall_s,
    }

    full_report = {
        "run_id":    run_id,
        "target":    target,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hw":        hw,
        "sw":        sw,
        "tests":     records,
        "summary":   summary,
    }

    # ── Write JSON benchmark ──────────────────────────────────────────────────
    benchmarks_dir = os.path.join(out_dir, "benchmarks")
    os.makedirs(benchmarks_dir, exist_ok=True)

    bench_path = os.path.join(benchmarks_dir, f"{target}_{gpu_slug}_{run_id}.json")
    with open(bench_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n  benchmark saved → {bench_path}", file=sys.stderr)

    # ── Print RTF table to stdout ─────────────────────────────────────────────
    print(render_table(records, hw, sw, summary))

    sys.exit(0 if fail_cnt == 0 else 1)


if __name__ == "__main__":
    main()
