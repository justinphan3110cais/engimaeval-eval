#!/usr/bin/env python3
"""
Export EnigmaEval dataset from HuggingFace to a local pickle file.

This script requires:
1. HF_TOKEN environment variable set in .env file
2. Access to the private cais/enigmaeval dataset on HuggingFace

Usage:
    python export_dataset.py
"""

import pickle
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset

# Load environment variables
load_dotenv()

def export_enigmaeval_dataset(output_path: str = "data/enigmaeval.pkl"):
    """
    Download and export the EnigmaEval dataset to a pickle file.
    
    Args:
        output_path: Path where the pickle file will be saved
    """
    print("="*60)
    print("EnigmaEval Dataset Export")
    print("="*60)
    
    # Create data directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from HuggingFace (requires HF_TOKEN)
    print("\n📥 Loading dataset from HuggingFace...")
    print("   Dataset: cais/enigmaeval")
    print("   Note: This requires HF_TOKEN in your .env file")
    
    try:
        dataset = load_dataset("cais/enigmaeval", split="test")
        print(f"✓ Successfully loaded {len(dataset)} samples")
        
        # Export to pickle
        print(f"\n💾 Exporting to {output_path}...")
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        # Get file size
        file_size = output_file.stat().st_size / (1024 * 1024)  # MB
        print(f"✓ Dataset exported successfully!")
        print(f"   File size: {file_size:.2f} MB")
        print(f"   Location: {output_file.absolute()}")
        
        # Verify the export
        print(f"\n🔍 Verifying export...")
        with open(output_file, 'rb') as f:
            loaded_data = pickle.load(f)
        print(f"✓ Verification successful! Loaded {len(loaded_data)} samples")
        
        print("\n" + "="*60)
        print("✅ Export completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. The dataset is saved in data/enigmaeval.pkl")
        print("2. This file is gitignored and won't be committed")
        print("3. Run enigmaeval_eval.py to evaluate models")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure HF_TOKEN is set in your .env file")
        print("2. Verify you have access to cais/enigmaeval dataset")
        print("3. Check your internet connection")
        raise

if __name__ == "__main__":
    export_enigmaeval_dataset()

