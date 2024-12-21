import requests
import gzip
import shutil
import os

# Download the file
url = "https://storage.googleapis.com/chesspic/datasets/2021-07-31-lichess-evaluations-37MM.db.gz"
filename = "2021-07-31-lichess-evaluations-37MM.db.gz"
response = requests.get(url, stream=True)

# Save the downloaded file
with open(filename, 'wb') as f:
    shutil.copyfileobj(response.raw, f)

# Decompress the .gz file
with gzip.open(filename, 'rb') as f_in:
    with open("2021-07-31-lichess-evaluations-37MM.db", 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

os.remove(filename)
print("Download and decompression complete.")