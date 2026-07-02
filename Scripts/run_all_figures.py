#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master script - Runs all Figure_*.py scripts
"""

from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent

# Find all Figure_*.py files
figure_scripts = sorted(SCRIPT_DIR.glob("Figure_*.py"))

print(f"Found {len(figure_scripts)} figure scripts.\n")

for script_path in figure_scripts:
    print(f"▶ Running {script_path.name} ...")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)], 
            capture_output=True, 
            text=True, 
            cwd=SCRIPT_DIR,
            timeout=300  # 5 minutes max per script
        )
        
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print("Errors:", result.stderr.strip())
        
        print(f"   ✓ {script_path.name} completed\n")
    except subprocess.TimeoutExpired:
        print(f"   ✗ Timeout running {script_path.name}\n")
    except Exception as e:
        print(f"   ✗ Error running {script_path.name}: {e}\n")

print("🎉 All figures processing completed!")
