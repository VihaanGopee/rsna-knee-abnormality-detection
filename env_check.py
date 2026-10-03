#!/usr/bin/env python3
"""Analyze the runtime environment for Phase 1 labeling.

Checks: CPU, RAM, GPU, Ollama status/models, Python deps.
Usage: python3 env_check.py
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import urllib.error


def run(cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:
        return ""


def check_cpu():
    info = {"cores": os.cpu_count(), "model": "unknown", "arch": platform.machine()}
    if sys.platform == "darwin":
        out = run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            info["model"] = out
        # Apple Silicon check
        if "Apple" in platform.processor() or info["arch"] == "arm64":
            info["apple_silicon"] = True
    elif sys.platform.startswith("linux"):
        out = run(["grep", "-m1", "model name", "/proc/cpuinfo"])
        if ":" in out:
            info["model"] = out.split(":", 1)[1].strip()
    return info


def check_ram():
    info = {}
    try:
        if sys.platform == "darwin":
            total = int(run(["sysctl", "-n", "hw.memsize"])) / 1e9
            info["total_gb"] = round(total, 1)
            # available via vm_stat (rough)
            out = run(["vm_stat"])
            info["note"] = "see Activity Monitor for free RAM"
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info["total_gb"] = round(int(line.split()[1]) / 1e6, 1)
                    elif line.startswith("MemAvailable:"):
                        info["avail_gb"] = round(int(line.split()[1]) / 1e6, 1)
    except Exception as e:
        info["error"] = str(e)
    return info


def check_gpu():
    gpus = []
    # NVIDIA
    out = run(["nvidia-smi", "--query-gpu=name,memory.total",
               "--format=csv,noheader"])
    if out and "failed" not in out.lower():
        for line in out.split("\n"):
            if line.strip():
                gpus.append({"type": "nvidia", "info": line.strip()})
    # Apple Silicon (unified memory = GPU shares RAM)
    if sys.platform == "darwin" and platform.machine() == "arm64":
        gpus.append({"type": "apple_silicon",
                     "info": "unified memory GPU (shares system RAM)"})
    # macOS discrete GPU info
    if sys.platform == "darwin":
        out = run(["system_profiler", "SPDisplaysDataType"])
        if "Chipset Model" in out:
            for line in out.split("\n"):
                if "Chipset Model" in line:
                    gpus.append({"type": "mac_display",
                                 "info": line.split(":")[1].strip()})
                    break
    return gpus


def check_ollama():
    info = {"running": False, "models": []}
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.load(resp)
            info["running"] = True
            for m in data.get("models", []):
                info["models"].append({
                    "name": m["name"],
                    "size_gb": round(m.get("size", 0) / 1e9, 1),
                })
    except Exception as e:
        info["error"] = f"not reachable: {e}"
    return info


def check_python_deps():
    deps = {}
    for pkg in ["pandas", "numpy", "requests"]:
        try:
            mod = __import__(pkg)
            deps[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            deps[pkg] = "MISSING"
    return deps


def main():
    print("=" * 60)
    print("ENVIRONMENT CHECK for Phase 1 labeling")
    print("=" * 60)

    cpu = check_cpu()
    print(f"\nCPU: {cpu['model']}")
    print(f"  cores: {cpu['cores']} | arch: {cpu['arch']}")

    ram = check_ram()
    print(f"\nRAM: {ram.get('total_gb', '?')} GB total", end="")
    if "avail_gb" in ram:
        print(f" | {ram['avail_gb']} GB available")
    else:
        print()

    gpus = check_gpu()
    print(f"\nGPU:")
    if gpus:
        for g in gpus:
            print(f"  [{g['type']}] {g['info']}")
    else:
        print("  none detected")

    oll = check_ollama()
    print(f"\nOllama: {'RUNNING' if oll['running'] else 'NOT RUNNING'}")
    for m in oll["models"]:
        print(f"  {m['name']} ({m['size_gb']} GB)")

    deps = check_python_deps()
    print(f"\nPython {platform.python_version()}:")
    for k, v in deps.items():
        print(f"  {k}: {v}")

    # verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    ram_gb = ram.get("total_gb", 0)
    has_14b = any("14b" in m["name"] for m in oll["models"])
    if oll["running"] and has_14b and ram_gb >= 12:
        print("✓ Ready for phase1_mac_batch.py (qwen3:14b via Ollama)")
    elif oll["running"] and oll["models"]:
        print(f"! Ollama running but no 14b model. Available: "
              f"{[m['name'] for m in oll['models']]}")
        print("  Run: ollama pull qwen3:14b")
    elif not oll["running"]:
        print("✗ Ollama not running. Start it: ollama serve")
    if ram_gb < 12 and ram_gb > 0:
        print(f"! Only {ram_gb} GB RAM — 14b needs ~10GB. May OOM.")
    missing = [k for k, v in deps.items() if v == "MISSING"]
    if missing:
        print(f"! Missing Python packages: {missing}. Run: pip install {' '.join(missing)}")


if __name__ == "__main__":
    main()
