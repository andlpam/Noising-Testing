import os
import shutil
import subprocess
"""Clean directory content"""
def create_clean_dirs(dir_path):
    if os.path.exists(dir_path):
        print(f"-> Cleaning dir: {dir_path}")
        
        # Se for Windows ('nt')
        if os.name == 'nt':
            # ignore erros if it is windows
            shutil.rmtree(dir_path, ignore_errors=True) 
        # if docker or linux Linux/Docker ('posix')
        else:
            #Use linux
            subprocess.run(['rm', '-rf', dir_path], check=False)
            
    os.makedirs(dir_path, exist_ok=True)
        
def turn_relative_path_into_full(rel_path, parent_path_full):
    dir_name = os.path.basename(rel_path)
    full_path = os.path.join(parent_path_full, dir_name)
    return full_path