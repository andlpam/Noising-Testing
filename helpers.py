import os
import shutil
"""Clean directory content"""
def create_clean_dirs(dir):
        if os.path.exists(dir):
            print(f"-> Cleaning dir: {dir}")
            shutil.rmtree(dir)
        
        os.makedirs(dir, exist_ok=True)
        
def turn_relative_path_into_full(rel_path, parent_path_full):
    dir_name = os.path.basename(rel_path)
    full_path = os.path.join(parent_path_full, dir_name)
    return full_path