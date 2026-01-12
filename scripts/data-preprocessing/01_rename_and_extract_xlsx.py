from pathlib import Path
import shutil

BASE_PATH = "./data/01-raw"
COPY_PATH = "./data/02-raw"

CURRENT_YEAR = "2025"

client_name = BASE_PATH.name

def rename_xlsx(folder: Path, year: str):
    for file in folder.glob("*.xlsx"):
        new_name = f"{year} - {file.stem}.xlsx"
        new_path = file.with_name(new_name)
        if not new_path.exists():
            file.rename(new_path)
        else:
            print(f"⚠️ Skipped (already exists): {new_path.name}")

def copy_xlsx(src_folder: Path, dest_folder: Path):
    for file in src_folder.glob("*.xlsx"):
        dest_file = dest_folder / file.name

        if not dest_file.exists():
            shutil.copy2(file, dest_file)
        else:
            print(f"⚠️ Skipped (already exists): {dest_file.name}")

for subfolder in BASE_PATH.iterdir():
    rename_xlsx(subfolder, CURRENT_YEAR)
    for subfolder_2 in subfolder.iterdir():
        if subfolder_2.is_dir():
            rename_xlsx(subfolder_2, subfolder_2.name)

for subfolder in BASE_PATH.iterdir():
    copy_xlsx(subfolder, COPY_PATH)
    for subfolder_2 in subfolder.iterdir():
        if subfolder_2.is_dir():
            copy_xlsx(subfolder_2, COPY_PATH)