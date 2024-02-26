import shlex
import subprocess
from typing import List


def get_fonts() -> List[str]:
    base_fonts = [
        "Arial",
        "Courier New",
        "Times New Roman",
        "Georgia",
        "Trebuchet MS",
        "Verdana",
        "Tex Gyre Schola Regular",
        "Noto Sans",
        "Noto Serif",
        "Lato",
        "DejaVu Sans",
        "DejaVu Serif",
        "FreeSans",
        "Fira Code",
        "Nimbus Sans",
        "Quicksand",
        "Liberation Sans",
        "Courier",
        "URW Gothic",
        "Comic Neue",
        "Cantarell",
    ]
    fonts = set()
    for font in base_fonts:
        result = subprocess.run(
            f"fc-match '{font}'", shell=True, check=True, capture_output=True, text=True
        )
        alternative_font = shlex.split(result.stdout)
        if len(alternative_font) > 1:
            fonts.add(alternative_font[-2])
    return list(fonts)


if __name__ == "__main__":
    print(get_fonts())
