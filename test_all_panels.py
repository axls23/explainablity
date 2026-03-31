#!/usr/bin/env python3
"""Test if Entropy, VAR, and Perturbation panels are all updating."""

import requests
import time
import sys

BASE_URL = "http://127.0.0.1:8000"

def check_panel_state(panel_name, endpoint):
    """Check state of a specific panel via debug endpoint."""
    try:
        resp = requests.get(f"{BASE_URL}/debug/{endpoint}", timeout=10)
        data = resp.json()
        return data
    except Exception as e:
        print(f"✗ Error checking {panel_name}: {e}")
        return None

def main():
    print("=" * 70)
    print("CHRONOSCOPE PANEL STATE TEST (Entropy, VAR, Perturbation)")
    print("=" * 70)
    
    # Health check
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✓ Server health: {resp.status_code}\n")
    except Exception as e:
        print(f"✗ Server not responding: {e}")
        sys.exit(1)
    
    # Check panel states BEFORE chat
    print("BEFORE chat:")
    print("-" * 70)
    
    metrics_before = check_panel_state("Entropy", "metrics")
    if metrics_before:
        entropy_count = sum(metrics_before.get("metrics_summary", {}).values())
        print(f"  Entropy: {entropy_count} total timesteps captured")
        print(f"    entropy_row present: {metrics_before.get('last_frame_has_entropy_row')}")
    
    pert_before = check_panel_state("Perturbation", "perturbation")
    if pert_before:
        print(f"  Perturbation: {pert_before.get('perturbation_count', 0)} results in frame")
    
    # Send chat WITH VAR AND INTERVENTION enabled
    print(f"\n→ Sending /chat request (run_var=true, run_intervention=true)...")
    print("-" * 70)
    
    payload = {
        "prompt": "Explain how attention mechanisms work in transformers",
        "capture_attentions": True,
        "run_var": True,           # Enable VAR analysis
        "run_intervention": True,  # Enable perturbation/ablation
        "max_new_tokens": 30
    }
    
    start = time.time()
    try:
        resp = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            timeout=300
        )
        elapsed = time.time() - start
        data = resp.json()
        tokens = data.get("tokens_generated", 0)
        print(f"✓ Chat completed in {elapsed:.1f}s")
        print(f"  Generated {tokens} tokens")
        print(f"  Status: {data.get('status')}")
        if data.get('response'):
            print(f"  Response: {data['response'][:80]}...")
    except Exception as e:
        print(f"✗ Chat failed: {e}")
        return
    
    # Wait for background analysis to complete
    print(f"\n⏳ Waiting for background analysis (VAR, HMM, Perturbation)...")
    time.sleep(5)
    
    # Check panel states AFTER chat
    print(f"\nAFTER chat:")
    print("-" * 70)
    
    metrics_after = check_panel_state("Entropy", "metrics")
    if metrics_after:
        entropy_count = sum(metrics_after.get("metrics_summary", {}).values())
        entropy_delta = entropy_count - sum(metrics_before.get("metrics_summary", {}).values()) if metrics_before else 0
        print(f"  Entropy: {entropy_count} total timesteps ({entropy_delta} new)")
        print(f"    entropy_row present: {metrics_after.get('last_frame_has_entropy_row')}")
        if entropy_delta == 0:
            print(f"    ⚠ WARNING: No new entropy data captured!")
    
    pert_after = check_panel_state("Perturbation", "perturbation")
    if pert_after:
        pert_count = pert_after.get('perturbation_count', 0)
        pert_delta = pert_count - (pert_before.get('perturbation_count', 0) if pert_before else 0)
        print(f"  Perturbation: {pert_count} results in frame ({pert_delta} new)")
        if pert_delta > 0:
            print(f"    ✓ Perturbation data IS updating")
            first_pert = pert_after.get('first_pert')
            if first_pert:
                print(f"    Sample: head={first_pert.get('head')}, delta_entropy={first_pert.get('delta_entropy'):.3f}")
        else:
            print(f"    ✗ NO perturbation data in frame")
            print(f"      Possible reasons:")
            print(f"        1. No significant VAR pairs found")
            print(f"        2. run_intervention not enabled")
            print(f"        3. Background analysis still running")
    
    # Check last frame keys to see what was actually pushed
    if pert_after:
        last_keys = pert_after.get("last_frame_keys", [])
        print(f"\n  Last frame contains these keys: {last_keys}")
        if "perturbation_results" not in last_keys:
            print(f"  ✗ 'perturbation_results' not in frame!")
    
    print(f"\n" + "=" * 70)
    print("DIAGNOSIS:")
    if pert_after and pert_after.get('perturbation_count', 0) > 0:
        print("✓ ALL PANELS updating correctly!")
    else:
        print("✗ Perturbation panel still idle")
        print("   Next: Check /debug/last_frame for full frame content")

if __name__ == "__main__":
    main()
