from pathlib import Path
import shutil

# PEP 8: Constants at the top
BASE_PATH = Path("./data/01-raw")
PROCESSED_PATH = Path("./data/02-interim")
CURRENT_YEAR = "2025"


# PEP 8: Two blank lines before top-level functions 
def process_and_copy_file(file_path: Path, dest_folder: Path, year: str = None):
    """
    Copies a file to the destination with a new name.
    Does NOT modify the source file.
    """
    # Fix naming logic
    if year:
        new_name = f"{year} - {file_path.name}"
    else:
        new_name = file_path.name

    dest_path = dest_folder / new_name

    if not dest_path.exists():
        # Create destination folder if it doesn't exist
        dest_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest_path)
        print(f"✅ Processed: {new_name}")
    else:
        print(f"⚠️ Skipped (exists): {new_name}")


def main():
    # Loop through the structure once
    for subfolder in BASE_PATH.iterdir():
        if subfolder.is_dir():
            # Level 1: Add Year
            for file in subfolder.glob("*.xlsx"):
                process_and_copy_file(file, PROCESSED_PATH, year=CURRENT_YEAR)
            
            # Level 2: Keep original name
            for subfolder_2 in subfolder.iterdir():
                if subfolder_2.is_dir():
                    for file in subfolder_2.glob("*.xlsx"):
                        process_and_copy_file(file, PROCESSED_PATH, year=subfolder_2.name)

if __name__ == "__main__":
    main()