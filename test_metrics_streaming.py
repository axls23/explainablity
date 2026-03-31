#!/usr/bin/env python3
"""Diagnostic script to test if entropy/VAR metrics flow during streaming."""

import requests
import json
import time
import sys

BASE_URL = "http://127.0.0.1:8000"

def check_metrics_before():
    """Check metrics state before chat."""
    resp = requests.get(f"{BASE_URL}/debug/metrics", timeout=10)
    data = resp.json()
    summary = data.get("metrics_summary", {})
    if summary:
        total = sum(summary.values())
        print(f"✓ Metrics before chat: {len(summary)} layers, {total} total timesteps")
        return total
    else:
        print("✗ No metrics captured yet")
        return 0

def send_chat():
    """Send a test chat request."""
    payload = {
        "prompt": "What is entropy in information theory?",
        "capture_attentions": True,
        "run_var": False,
        "max_new_tokens": 20
    }
    print("\n→ Sending /chat request...")
    start = time.time()
    resp = requests.post(
        f"{BASE_URL}/chat",
        json=payload,
        timeout=120
    )
    elapsed = time.time() - start
    data = resp.json()
    tokens = data.get("tokens_generated", 0)
    print(f"✓ Chat completed in {elapsed:.1f}s, generated {tokens} tokens")
    print(f"  Response: {data.get('response', '')[:60]}...")
    return tokens

def check_metrics_after():
    """Check metrics state after chat."""
    resp = requests.get(f"{BASE_URL}/debug/metrics", timeout=10)
    data = resp.json()
    summary = data.get("metrics_summary", {})
    entropy_row_present = data.get("last_frame_entropy_row_is_none", None)
    
    if summary:
        total = sum(summary.values())
        print(f"\n✓ Metrics after chat: {len(summary)} layers, {total} total timesteps")
        print(f"  Last frame entropy_row is_none: {entropy_row_present}")
        
        # Check if entropy_row is actually in the frame
        has_entropy = data.get("last_frame_has_entropy_row", False)
        print(f"  Last frame has entropy_row field: {has_entropy}")
        
        return total
    else:
        print("\n✗ No metrics captured")
        return 0

def main():
    print("=" * 60)
    print("CHRONOSCOPE ENTROPY/VAR METRICS DIAGNOSTIC TEST")
    print("=" * 60)
    
    # Check health
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✓ Server health: {resp.status_code}\n")
    except Exception as e:
        print(f"✗ Server not responding: {e}")
        sys.exit(1)
    
    # Check metrics before
    before = check_metrics_before()
    
    # Send chat
    tokens = send_chat()
    
    # Give it a moment to process
    time.sleep(1)
    
    # Check metrics after
    after = check_metrics_after()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Timesteps before: {before}")
    print(f"  Timesteps after: {after}")
    print(f"  New timesteps: {after - before}")
    print(f"  Expected ~{tokens} new timesteps per layer")
    
    if after > before:
        print("\n✓ Metrics ARE being captured and flowing!")
        print("  The entropy and VAR panels should be updating on the dashboard.")
        print("  If they aren't, check if the dashboard is connected to the WebSocket.")
    else:
        print("\n✗ No new metrics captured - metrics pipeline may be broken")

if __name__ == "__main__":
    main()
