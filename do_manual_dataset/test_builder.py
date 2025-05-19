# filepath: d:\Malak Doc\Malak Education\MBZUAI\Academic years\Spring 2025\ICL\DO_manual_rlds_builder\do_manual_dataset\test_builder.py
import sys
import os
# Set the Python path to include the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Now import directly from the module
from do_manual_dataset import DoManualDataset

if __name__ == "__main__":
    print("Testing builder...")
    builder = DoManualDataset()
    print(f"Builder: {builder}")
    print(f"Builder info: {builder.info}")
    print("Successfully created builder!")