"""Japanese font selection for reproducible v5-style scientific figures."""
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager

def apply_style(root):
    configured=os.environ.get('ICA_JAPANESE_FONT')
    candidates=([Path(configured)] if configured else [])+[
        root/'fonts/NotoSansCJKjp-Regular.otf',
        Path('C:/Windows/Fonts/meiryo.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
    ]
    font=next((p for p in candidates if p.is_file()),None)
    if font is None:
        raise FileNotFoundError('Install a Japanese font or set ICA_JAPANESE_FONT to its file path.')
    fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':FontProperties(fname=str(font)).get_name(),
                         'axes.unicode_minus':False,'font.size':11,'pdf.fonttype':42})
