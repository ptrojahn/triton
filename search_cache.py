import os

def search_files(folder_path):
    # Loop through the directory and subdirectories
    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            # Check if the file content includes "_swiglu"
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    lines = file.readlines()
                    if any('_swiglu,@function' in line for line in lines):
                        # Check each line for "vgpr_count"
                        print(f"File: {root}/{filename}")
                        for line in lines:
                            if '_swiglu,@function' in line:
                                print(f"{line.strip()}")
                            elif '.vgpr_count' in line:
                                print(f"{line.strip()}")
                            elif '.sgpr_count' in line:
                                print(f"{line.strip()}")
                            elif 'codeLenInByte' in line:
                                print(f"{line.strip()}")
            except Exception as e:
                print(f"An error occurred while processing {file_path}: {e}")

# Define the folder to search
folder_path = os.path.expanduser("~") + '/.triton/cache/'
search_files(folder_path)