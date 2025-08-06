#!/usr/bin/env python3
"""
MAC_Bench Dataset Download Script

Downloads the MAC_Bench dataset from Hugging Face Hub to the local directory.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def download_dataset():
    """Download MAC_Bench dataset from Hugging Face"""
    
    print("🔄 Starting dataset download...")
    print("📡 Source: https://huggingface.co/datasets/mhjiang0408/MAC_Bench")
    
    target_dir = "./MAC_Bench"
    
    # Check if dataset already exists
    if os.path.exists(target_dir):
        print(f"⚠️  Dataset directory '{target_dir}' already exists")
        response = input("Do you want to re-download? [y/N]: ")
        if response.lower() != 'y':
            print("✅ Skipping dataset download")
            return True
        
        # Remove existing directory
        import shutil
        shutil.rmtree(target_dir)
        print("🗑️  Removed existing dataset directory")
    
    try:
        print("📥 Downloading dataset... (this may take several minutes)")
        snapshot_download(
            repo_id="mhjiang0408/MAC_Bench",
            repo_type="dataset", 
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        
        # Check download success
        if os.path.exists(target_dir):
            # Get directory size
            def get_dir_size(path):
                total = 0
                for dirpath, dirnames, filenames in os.walk(path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total += os.path.getsize(filepath)
                return total
            
            size_bytes = get_dir_size(target_dir)
            size_mb = size_bytes / (1024 * 1024)
            
            print(f"✅ Dataset downloaded successfully!")
            print(f"📁 Location: {target_dir}")
            print(f"📊 Size: {size_mb:.1f} MB")
            return True
        else:
            print("❌ Dataset download failed - directory not created")
            return False
            
    except Exception as e:
        print(f"❌ Dataset download failed: {str(e)}")
        print("💡 Possible solutions:")
        print("   1. Check your internet connection")
        print("   2. Try: huggingface-cli login")
        print("   3. Install missing dependencies: pip install huggingface_hub")
        return False

if __name__ == "__main__":
    success = download_dataset()
    sys.exit(0 if success else 1)