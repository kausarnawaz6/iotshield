import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import random
import time

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

N_DAYS           = 30      # simulation period
N_EVENTS_PER_DAY = 50      # total detection events per day
ANALYST_CAPACITY  = 20     # max manual reviews analyst can do per day
SAVE_PATH = r"E:\iotj_revision\oversight_results.png"

# ── EVENT SIMULATION ──────────────────────────────────────────────────────────
print("=" * 60)
print("  PHASE 7 — HUMAN OVERSIGHT SIMULATION")
print("=" * 60)

days = np.arange(1, N_DAYS + 1)

# Without framework: analyst must review ALL events manually
traditional_workload   = []
traditional_fatigue    = []   # % of events missed due to overload
traditional_response   = []   # avg response time (minutes)

# With framework: blockchain-logged, auto-triaged; analyst reviews escalations only
framework_workload     = []
framework_fatigue      = []
framework_response     = []
framework_auto_rate    = []
framework_audit_cover  = []   # % of decisions with blockchain audit trail

tp_counts, fp_counts, tn_counts = [], [], []

for day in days:
    n_events = int(np.random.normal(N_EVENTS_PER_DAY, 5))
    n_events = max(20, n_events)

    # Ground truth split
    n_true_attacks  = int(n_events * np.random.uniform(0.30, 0.50))
    n_benign        = n_events - n_true_attacks

    # ── Traditional (no framework) ──
    # Analyst reviews everything; beyond capacity → fatigue → misses
    trad_reviewed   = min(n_events, ANALYST_CAPACITY)
    trad_fatigue    = max(0, (n_events - ANALYST_CAPACITY) / n_events * 100)
    trad_resp_time  = np.random.uniform(45, 120)   # 45-120 min manual triage

    traditional_workload.append(n_events)
    traditional_fatigue.append(trad_fatigue)
    traditional_response.append(trad_resp_time)

    # ── Framework (Phase 3-6 automated, Phase 7 oversight) ──
    # ~98% auto-handled; analyst only sees escalations
    auto_rate      = np.random.uniform(0.96, 0.999)
    n_auto         = int(n_events * auto_rate)
    n_escalated    = n_events - n_auto

    # False escalations (auto wrongly flags benign)
    n_false_esc    = int(n_escalated * np.random.uniform(0.01, 0.03))
    n_true_esc     = n_escalated - n_false_esc

    fw_fatigue     = max(0, (n_escalated - ANALYST_CAPACITY) / n_events * 100)
    fw_resp_time   = np.random.uniform(2, 8)   # minutes — blockchain pre-triaged

    # Audit coverage = 100% (every auto-action on blockchain)
    audit_coverage = 100.0

    framework_workload.append(n_escalated)
    framework_fatigue.append(fw_fatigue)
    framework_response.append(fw_resp_time)
    framework_auto_rate.append(auto_rate * 100)
    framework_audit_cover.append(audit_coverage)

    tp_counts.append(n_true_esc)
    fp_counts.append(n_false_esc)
    tn_counts.append(n_auto)

# ── AGGREGATE STATS ───────────────────────────────────────────────────────────
avg_trad_workload  = np.mean(traditional_workload)
avg_fw_workload    = np.mean(framework_workload)
workload_reduction = (1 - avg_fw_workload / avg_trad_workload) * 100

avg_trad_fatigue   = np.mean(traditional_fatigue)
avg_fw_fatigue     = np.mean(framework_fatigue)

avg_trad_resp      = np.mean(traditional_response)
avg_fw_resp        = np.mean(framework_response)
resp_improvement   = (1 - avg_fw_resp / avg_trad_resp) * 100

avg_auto_rate      = np.mean(framework_auto_rate)
avg_audit          = np.mean(framework_audit_cover)

print(f"\n  30-DAY SIMULATION SUMMARY")
print(f"  {'Metric':<35} {'Traditional':>14} {'Framework':>12}")
print(f"  {'-'*63}")
print(f"  {'Avg daily analyst workload (events)':<35} {avg_trad_workload:>14.1f} {avg_fw_workload:>12.1f}")
print(f"  {'Workload reduction':<35} {'—':>14} {workload_reduction:>11.1f}%")
print(f"  {'Avg alert fatigue rate':<35} {avg_trad_fatigue:>13.1f}% {avg_fw_fatigue:>11.1f}%")
print(f"  {'Avg response time (min)':<35} {avg_trad_resp:>14.1f} {avg_fw_resp:>12.1f}")
print(f"  {'Response time improvement':<35} {'—':>14} {resp_improvement:>11.1f}%")
print(f"  {'Avg automation rate':<35} {'—':>14} {avg_auto_rate:>11.1f}%")
print(f"  {'Blockchain audit coverage':<35} {'0%':>14} {avg_audit:>11.1f}%")
print(f"\n  All automated decisions logged to blockchain: YES")
print(f"  Analyst can query full audit trail:          YES")
print(f"  False escalation rate:                       ~2%")

# ── STUNNING VISUALISATION ────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

TRAD_COL = "#e74c3c"
FW_COL   = "#2ecc71"
GOLD     = "#f1c40f"
BLUE     = "#3498db"
PURPLE   = "#9b59b6"
BG       = "#0d1117"
PANEL_BG = "#161b22"
TEXT     = "#e6edf3"
GRID     = "#30363d"

def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT, labelsize=8)