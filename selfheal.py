import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

N_NODES          = 312      # matches your SUS paper scenario
N_COMPROMISED    = 47       # ~15% nodes compromised (realistic breach)
N_SIMULATIONS    = 50       # Monte Carlo runs for statistical confidence
HEAL_TIMEOUT_MIN = 30       # max minutes for automated healing
SAVE_PATH        = r"E:\iotj_revision\selfheal_results.png"

# ── DATA STRUCTURES ───────────────────────────────────────────────────────────
@dataclass
class IoTNode:
    node_id:        int
    device_type:    str
    firmware_hash:  str        # golden hash from blockchain snapshot
    puf_response:   str        # hardware fingerprint (simulated)
    baseline_behavior: np.ndarray  # normal traffic feature vector
    is_compromised: bool = False
    is_healed:      bool = False
    heal_time_min:  float = 0.0
    heal_method:    str = ""

@dataclass
class DigitalTwin:
    node_id:    int
    snapshots:  List[Dict] = field(default_factory=list)

    def add_snapshot(self, firmware_hash, behavior, timestamp):
        self.snapshots.append({
            "firmware_hash": firmware_hash,
            "behavior":      behavior,
            "timestamp":     timestamp,
            "block_hash":    hashlib.sha256(
                f"{firmware_hash}{timestamp}".encode()).hexdigest()
        })

    def get_safest_snapshot(self, current_behavior):
        """Return snapshot with minimum behavioral drift from current baseline"""
        if not self.snapshots:
            return None
        drifts = [
            np.linalg.norm(s["behavior"] - current_behavior)
            for s in self.snapshots
        ]
        return self.snapshots[np.argmin(drifts)]

# ── SIMULATION HELPERS ────────────────────────────────────────────────────────
DEVICE_TYPES = [
    "Smart_Camera", "IoT_Gateway", "Temperature_Sensor",
    "Smart_Meter",  "Medical_Monitor", "Industrial_PLC",
    "Smart_Lock",   "HVAC_Controller"
]

def generate_node(node_id):
    firmware = hashlib.sha256(f"firmware_v2.1_{node_id}".encode()).hexdigest()
    puf      = hashlib.sha256(f"puf_secret_{node_id}_{random.randint(0,99999)}".encode()).hexdigest()
    behavior = np.random.normal(0.5, 0.1, 115)   # 115-dim baseline (matches N-BaIoT feature space)
    return IoTNode(
        node_id        = node_id,
        device_type    = random.choice(DEVICE_TYPES),
        firmware_hash  = firmware,
        puf_response   = puf,
        baseline_behavior = behavior
    )

def build_digital_twin(node, n_snapshots=5):
    twin = DigitalTwin(node_id=node.node_id)
    for i in range(n_snapshots):
        # Snapshots slightly vary — simulating firmware updates over time
        snap_behavior = node.baseline_behavior + np.random.normal(0, 0.01, 115)
        snap_firmware = hashlib.sha256(
            f"firmware_v2.{i}_{node.node_id}".encode()).hexdigest()
        twin.add_snapshot(snap_firmware, snap_behavior, timestamp=i*86400)
    return twin

def simulate_compromise(node):
    """Attacker modifies firmware and behavior"""
    node.is_compromised     = True
    node.firmware_hash      = hashlib.sha256(b"malicious_firmware").hexdigest()
    node.baseline_behavior  = node.baseline_behavior + np.random.normal(0.3, 0.15, 115)
    return node

def verify_puf(node, expected_puf):
    """PUF hardware attestation — returns True if hardware matches"""
    # Simulate 2% PUF false-reject rate (hardware noise)
    if random.random() < 0.02:
        return False
    return node.puf_response == expected_puf

def attempt_heal(node, twin, original_puf, attempt_num):
    """
    Three-stage healing:
    1. SDN isolation (immediate)
    2. Digital twin rollback
    3. PUF reintegration
    Returns (success, time_minutes, method)
    """
    # Stage 1: SDN isolation — instant
    isolation_time = np.random.uniform(0.1, 0.5)

    # Stage 2: Digital twin identifies safest snapshot & rolls back
    snapshot = twin.get_safest_snapshot(node.baseline_behavior)
    if snapshot is None:
        return False, HEAL_TIMEOUT_MIN, "NO_SNAPSHOT"

    rollback_time = np.random.uniform(2.0, 8.0)   # minutes

    # Restore firmware from blockchain-verified snapshot
    node.firmware_hash     = snapshot["firmware_hash"]
    node.baseline_behavior = snapshot["behavior"]

    # Stage 3: PUF hardware attestation before reintegration
    puf_time = np.random.uniform(0.5, 2.0)

    if verify_puf(node, original_puf):
        total_time = isolation_time + rollback_time + puf_time
        node.is_healed   = True
        node.heal_time_min = total_time
        node.heal_method = "FULL_AUTO"
        return True, total_time, "FULL_AUTO"
    else:
        # PUF failed — escalate to manual review
        manual_time = np.random.uniform(15.0, 25.0)
        total_time  = isolation_time + rollback_time + puf_time + manual_time
        if total_time <= HEAL_TIMEOUT_MIN:
            node.is_healed   = True
            node.heal_time_min = total_time
            node.heal_method = "MANUAL_ESCALATION"
            return True, total_time, "MANUAL_ESCALATION"
        return False, total_time, "TIMEOUT"

# ── MONTE CARLO SIMULATION ────────────────────────────────────────────────────
print("=" * 58)
print("  PHASE 6 — SELF-HEALING RESPONSE SIMULATION")
print("=" * 58)
print(f"  Network size    : {N_NODES} nodes")
print(f"  Compromised     : {N_COMPROMISED} nodes ({N_COMPROMISED/N_NODES*100:.1f}%)")
print(f"  Monte Carlo runs: {N_SIMULATIONS}")
print(f"  Heal timeout    : {HEAL_TIMEOUT_MIN} minutes\n")

all_recovery_rates   = []
all_heal_times       = []
all_method_counts    = []
all_24h_rates        = []

for sim in range(N_SIMULATIONS):
    # Build network
    nodes  = [generate_node(i) for i in range(N_NODES)]
    twins  = {n.node_id: build_digital_twin(n) for n in nodes}
    pufs   = {n.node_id: n.puf_response for n in nodes}

    # Compromise random subset
    compromised_ids = random.sample(range(N_NODES), N_COMPROMISED)
    for cid in compromised_ids:
        simulate_compromise(nodes[cid])

    # Attempt healing for each compromised node
    healed       = 0
    heal_times   = []
    methods      = {"FULL_AUTO": 0, "MANUAL_ESCALATION": 0, "TIMEOUT": 0, "NO_SNAPSHOT": 0}

    for cid in compromised_ids:
        node    = nodes[cid]
        twin    = twins[cid]
        orig_puf = pufs[cid]

        success, t, method = attempt_heal(node, twin, orig_puf, sim)
        methods[method] = methods.get(method, 0) + 1

        if success:
            healed += 1
            heal_times.append(t)

    recovery_rate = healed / N_COMPROMISED * 100
    within_24h    = sum(1 for t in heal_times if t <= 1440) / N_COMPROMISED * 100

    all_recovery_rates.append(recovery_rate)
    all_heal_times.extend(heal_times)
    all_24h_rates.append(within_24h)
    all_method_counts.append(methods)

    if sim % 10 == 0:
        print(f"  Sim {sim+1:3d}/{N_SIMULATIONS} | "
              f"Recovery={recovery_rate:.1f}% | "
              f"Avg heal={np.mean(heal_times):.1f}min | "
              f"Auto={methods['FULL_AUTO']} Manual={methods['MANUAL_ESCALATION']}")

# ── AGGREGATE RESULTS ─────────────────────────────────────────────────────────
avg_recovery  = np.mean(all_recovery_rates)
std_recovery  = np.std(all_recovery_rates)
avg_heal_time = np.mean(all_heal_times)
std_heal_time = np.std(all_heal_times)
p95_heal_time = np.percentile(all_heal_times, 95)
avg_24h       = np.mean(all_24h_rates)

total_methods = {k: sum(d[k] for d in all_method_counts)
                 for k in ["FULL_AUTO","MANUAL_ESCALATION","TIMEOUT","NO_SNAPSHOT"]}
total_healed  = total_methods["FULL_AUTO"] + total_methods["MANUAL_ESCALATION"]
auto_pct      = total_methods["FULL_AUTO"]   / total_healed * 100 if total_healed else 0
manual_pct    = total_methods["MANUAL_ESCALATION"] / total_healed * 100 if total_healed else 0

print(f"\n{'='*58}")
print(f"  SELF-HEALING — FINAL RESULTS ({N_SIMULATIONS} Monte Carlo runs)")
print(f"{'='*58}")
print(f"  Recovery rate   : {avg_recovery:.1f}% ± {std_recovery:.1f}%")
print(f"  Avg heal time   : {avg_heal_time:.2f} ± {std_heal_time:.2f} min")
print(f"  P95 heal time   : {p95_heal_time:.2f} min")
print(f"  Within 24 hours : {avg_24h:.1f}%")
print(f"  Full auto heal  : {auto_pct:.1f}% of healed nodes")
print(f"  Manual escalated: {manual_pct:.1f}% of healed nodes")
print(f"  Timeouts        : {total_methods['TIMEOUT']} across all runs")
print(f"{'='*58}")

# ── PLOTS ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Recovery rate distribution across MC runs
axes[0].hist(all_recovery_rates, bins=15, color="steelblue",
             edgecolor="black", alpha=0.8)
axes[0].axvline(avg_recovery, color="red", lw=2,
                linestyle="--", label=f"Mean={avg_recovery:.1f}%")
axes[0].set_xlabel("Recovery Rate (%)")
axes[0].set_ylabel("Frequency (MC runs)")
axes[0].set_title(f"Recovery Rate Distribution\n({N_SIMULATIONS} Monte Carlo Runs)")
axes[0].legend()

# Plot 2: Heal time distribution
axes[1].hist(all_heal_times, bins=30, color="darkorange",
             edgecolor="black", alpha=0.8)
axes[1].axvline(avg_heal_time, color="red", lw=2,
                linestyle="--", label=f"Mean={avg_heal_time:.1f} min")
axes[1].axvline(p95_heal_time, color="purple", lw=2,
                linestyle=":", label=f"P95={p95_heal_time:.1f} min")
axes[1].set_xlabel("Heal Time (minutes)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Heal Time Distribution\n(All Healed Nodes, All Runs)")
axes[1].legend()

# Plot 3: Healing method breakdown
method_labels = ["Full Auto\n(PUF pass)", "Manual\nEscalation", "Timeout", "No Snapshot"]
method_values = [total_methods["FULL_AUTO"],
                 total_methods["MANUAL_ESCALATION"],
                 total_methods["TIMEOUT"],
                 total_methods["NO_SNAPSHOT"]]
colors_pie = ["#2ecc71","#f39c12","#e74c3c","#95a5a6"]
axes[2].pie(method_values, labels=method_labels, colors=colors_pie,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor":"white","linewidth":1.5})
axes[2].set_title("Healing Method Breakdown\n(Across All Monte Carlo Runs)")

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150)
plt.show()
print(f"\nPlots saved to {SAVE_PATH}")