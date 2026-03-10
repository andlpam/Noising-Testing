import os
import shutil
"""Clean directory content"""
def create_clean_dirs(dir):
        if os.path.exists(dir):
            print(f"-> Cleaning dir: {dir}")
            shutil.rmtree(dir)
        
        os.makedirs(dir, exist_ok=True)