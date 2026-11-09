
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ("https://repo1.maven.org/maven2/com/github/javaparser/javaparser-core/3.5.6/javaparser-core-3.5.6.jar",
     "myabs/lib/javaparser-core-3.5.6.jar"),
    ("https://repo1.maven.org/maven2/org/antlr/antlr4-runtime/4.5.3/antlr4-runtime-4.5.3.jar",
     "myabs/lib/antlr4-runtime-4.5.3.jar"),
    ("https://raw.githubusercontent.com/micheletufano/src2abs/master/lib/javalexer.jar",
     "myabs/lib/javalexer.jar"),
    ("https://raw.githubusercontent.com/micheletufano/src2abs/master/idioms/idioms.csv",
     "myabs/idioms/idioms.csv"),
]

if __name__ == "__main__":
    for url, dest in FILES:
        path = ROOT / dest
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            print("Already exists:", dest)
            continue
        print("Downloading:", dest)
        urllib.request.urlretrieve(url, path)
    print("Done.")
