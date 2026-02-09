import os

CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB

def split_file(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    file_size = os.path.getsize(filename)
    parts = (file_size // CHUNK_SIZE) + (1 if file_size % CHUNK_SIZE else 0)
    
    print(f"Splitting {filename} ({file_size / (1024*1024):.2f} MB) into {parts} parts...")

    with open(filename, 'rb') as f:
        for i in range(parts):
            chunk = f.read(CHUNK_SIZE)
            part_filename = f"{filename}.part{i}"
            with open(part_filename, 'wb') as part_file:
                part_file.write(chunk)
            print(f"Created {part_filename}")

    print(f"Successfully split {filename}!")

if __name__ == "__main__":
    split_file("similarity.pkl")
    split_file("vectors.pkl")
