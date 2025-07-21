from pathlib import Path
import requests

def download_mnist(path):
    url = "https://github.com/pytorch/tutorials/raw/main/_static/"
    filename = "mnist.pkl.gz"

    if not (path / filename).exists():
        content = requests.get(url + filename).content
        (path / filename).open("wb").write(content)
    
    return path / filename

data_path = Path("data") if Path("data").exists() else Path("../data")
path = data_path / "downloaded" / "vector-mnist"
path.mkdir(parent=True, exist_ok=True)

datafile = download_mnist(path)
