#!/usr/bin/env python3
"""
Download all Vistaar benchmark datasets
"""
import os
import subprocess
import sys
from pathlib import Path

# Benchmark downloads
BENCHMARKS = {
    'kathbath': 'https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/kathbath.zip',
    'kathbath_noisy': 'https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/kathbath_noisy.zip',
    'commonvoice': 'https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/commonvoice.zip',
    'fleurs': 'https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/fleurs.zip',
    'indictts': 'https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/indictts.zip',
    'mucs': 'https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/mucs.zip',
    'gramvaani': 'https://indicwhisper.objectstore.e2enetworks.net/vistaar_benchmarks/gramvaani.zip',
}

def download_and_extract(name, url, download_dir):
    """Download and extract a benchmark dataset"""
    zip_path = os.path.join(download_dir, f'{name}.zip')
    
    print(f"\n{'='*60}")
    print(f"Downloading {name}...")
    print(f"{'='*60}")
    
    # Download with curl
    cmd = ['curl', '-L', '--progress-bar', url, '-o', zip_path]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error downloading {name}")
        return False
    
    # Check if file exists and has size
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
        print(f"Error: Downloaded file is empty or doesn't exist")
        return False
    
    # Extract
    print(f"\nExtracting {name}...")
    cmd = ['unzip', '-q', zip_path, '-d', download_dir]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error extracting {name}")
        return False
    
    # Remove zip
    os.remove(zip_path)
    print(f"✓ {name} downloaded and extracted")
    return True

def main():
    # Create benchmarks directory
    download_dir = '/home/vistaar/benchmarks'
    os.makedirs(download_dir, exist_ok=True)
    
    print(f"Downloading Vistaar benchmarks to: {download_dir}")
    print(f"Total datasets: {len(BENCHMARKS)}")
    
    success = []
    failed = []
    
    for name, url in BENCHMARKS.items():
        try:
            if download_and_extract(name, url, download_dir):
                success.append(name)
            else:
                failed.append(name)
        except KeyboardInterrupt:
            print(f"\nInterrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"Exception: {e}")
            failed.append(name)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successfully downloaded: {len(success)}")
    for s in success:
        print(f"  - {s}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}")
        for f in failed:
            print(f"  - {f}")
    
    print(f"\nBenchmarks directory: {download_dir}")
    try:
        size = subprocess.check_output(['du', '-sh', download_dir]).decode().split()[0]
        print(f"Total size: {size}")
    except:
        pass

if __name__ == '__main__':
    main()
