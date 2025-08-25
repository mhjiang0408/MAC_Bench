#!/usr/bin/env python3
"""
MAC_Bench Dataset Download Script

Downloads the MAC_Bench dataset from Hugging Face Hub to the local directory.
Supports resume functionality for interrupted downloads.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def check_partial_download(target_dir):
    """Check if there's a partial download that can be resumed"""
    if not os.path.exists(target_dir):
        return False, 0
    
    # Check for .incomplete files or other indicators of partial download
    incomplete_files = []
    total_files = 0
    
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            total_files += 1
            if file.endswith('.incomplete') or file.startswith('.tmp'):
                incomplete_files.append(os.path.join(root, file))
    
    if incomplete_files:
        return True, total_files
    
    # Also check if directory exists but seems incomplete (very small or empty)
    if total_files < 5:  # Assume dataset should have at least 5 files
        return True, total_files
        
    return False, total_files

def download_dataset():
    """Download MAC_Bench dataset from Hugging Face with resume support"""
    
    print("🔄 Starting dataset download...")
    print("📡 Source: https://huggingface.co/datasets/mhjiang0408/MAC_Bench")
    
    target_dir = "./MAC_Bench"
    
    # Check for existing dataset or partial download
    is_partial, file_count = check_partial_download(target_dir)
    
    if os.path.exists(target_dir) and not is_partial:
        print(f"✅ Dataset directory '{target_dir}' already exists and appears complete")
        response = input("Do you want to re-download? [y/N]: ")
        if response.lower() != 'y':
            print("✅ Skipping dataset download")
            return True
        
        # Remove existing directory for fresh download
        import shutil
        shutil.rmtree(target_dir)
        print("🗑️  Removed existing dataset directory")
    elif is_partial:
        print(f"⚠️  Found partial download in '{target_dir}' ({file_count} files)")
        response = input("Do you want to resume the download? [Y/n]: ")
        if response.lower() in ['', 'y', 'yes']:
            print("🔄 Resuming download from where it left off...")
        else:
            # Remove existing directory for fresh download
            import shutil
            shutil.rmtree(target_dir)
            print("🗑️  Removed partial download, starting fresh")
    
    try:
        print("📥 Downloading dataset... (this may take several minutes)")
        print("💡 Download supports resume - you can safely interrupt and restart if needed")
        
        snapshot_download(
            repo_id="mhjiang0408/MAC_Bench",
            repo_type="dataset", 
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True  # Enable resume functionality
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