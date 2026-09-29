"""Matplotlib setup with a font that can render Ethiopic (Ge'ez) script."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ETHIOPIC_FONTS = [
    "/System/Library/Fonts/Supplemental/Kefa.ttc",
    "/System/Library/Fonts/Supplemental/KefaIII.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansEthiopic-Regular.ttf",
    "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf",
]


def setup_fonts():
    names = []
    for path in ETHIOPIC_FONTS:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            names.append(font_manager.FontProperties(fname=path).get_name())
    plt.rcParams["font.family"] = ["DejaVu Sans"] + names
    return names
