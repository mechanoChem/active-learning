import os
import shutil

def remove_pycache_dirs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == "__pycache__":
                pycache_dir = os.path.join(dirpath, dirname)
                print(f"Removing {pycache_dir}")
                shutil.rmtree(pycache_dir)

# Set the root directory from where you want to start removing __pycache__ directories
root_directory = "./"  # Replace with your directory path

remove_pycache_dirs(root_directory)