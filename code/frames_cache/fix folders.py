from pathlib import Path
import shutil

# 1. CHANGE THIS to the folder that contains all the "Video Project X" folders
base_dir = Path(r"C:\Users\gaiai\Downloads\Clipchamp")

# 2. This is where everything will be collected
output_dir =Path(r"C:\uni\cv\rps-people")

for sub in base_dir.iterdir():
    if sub.is_dir() and sub.name.startswith("Video Project"):
        # rglob('*') also catches files in deeper subfolders, if any
        for file in sub.rglob('*'):
            if file.is_file():
                # initial target name
                target = output_dir / file.name

                # if a file with the same name exists, add _1, _2, etc.
                if target.exists():
                    stem = file.stem
                    suffix = file.suffix
                    count = 1
                    while True:
                        new_name = f"{stem}_{count}{suffix}"
                        target = output_dir / new_name
                        if not target.exists():
                            break
                        count += 1

                # COPY the file
                shutil.copy2(file, target)
                # If you prefer to MOVE instead of copy, use:
                # shutil.move(str(file), str(target))
