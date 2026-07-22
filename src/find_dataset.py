import os

def find_files():
    print("Scanning workspace recursively for files...")
    for root, dirs, files in os.walk("."):
        # Exclude common directories to keep output clean
        if ".git" in root or ".venv" in root or "__pycache__" in root:
            continue
        for file in files:
            path = os.path.join(root, file)
            size = os.path.getsize(path)
            # Show file if it's not empty or if it's in a relevant directory
            print(f"{path} ({size} bytes)")

if __name__ == "__main__":
    find_files()
