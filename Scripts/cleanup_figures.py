#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleanup script - Removes all generated figures
"""

from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).resolve().parent.parent

folders_to_clean = [
    "Figure_1",
    "Figure_2",
    "Figure_3",
    "Figure_4",
]

print("🧹 Starting cleanup of all figures...\n")

for folder_name in folders_to_clean:
    folder_path = REPO_ROOT / folder_name
    if folder_path.exists():
        try:
            shutil.rmtree(folder_path)
            print(f"   Removed: {folder_name}/")
        except Exception as e:
            print(f"   Error removing {folder_name}: {e}")
    else:
        print(f"   Not found: {folder_name}/")

print("\n✅ Cleanup completed!")
