"""Vietnamese/English single-color preference -> canonical v3 color.
Copy as color_mapper.py beside color_normalization.py in System_ChatBox.
No substring matching; negation, ambiguous/multiple-color requests need clarification.
"""
import re
import unicodedata
from color_normalization import normalize_color

VI_TO_EN_COLOR = {
    'trắng':'White','đen':'Black','xám':'Gray','xám đậm':'Gray','xám nhạt':'Gray',
    'bạc':'Silver','xanh dương':'Blue','xanh da trời':'Blue','xanh nước biển':'Blue',
    'xanh navy':'Blue','xanh dương nhạt':'Blue','xanh dương đậm':'Blue',
    'đỏ':'Red','nâu':'Brown','nâu đậm':'Brown','nâu nhạt':'Brown',
    'trong suốt':'Clear','xanh lá':'Green','xanh lá cây':'Green','hồng':'Pink',
    'hồng nhạt':'Pink','hồng đậm':'Pink','vàng kim':'Gold','vàng ánh kim':'Gold',
    'be':'Beige','vàng':'Yellow','tím':'Purple','cam':'Orange','ngà':'Ivory',
    'trắng ngà':'Ivory','kem':'Cream','đồng':'Copper','đồng đỏ':'Copper','đồng thiếc':'Bronze',
    'xanh ngọc lam':'Turquoise','xanh mòng két':'Teal','vàng hồng':'Rose Gold',
    'nhiều màu':'Multicolor','đa sắc':'Multicolor','đa màu':'Multicolor',
}
def _normalize(text):
    text=unicodedata.normalize('NFKD',text.casefold().replace('đ','d'))
    return re.sub(r'\s+',' ',''.join(c for c in text if not unicodedata.combining(c))).strip()

LOOKUP={_normalize(k):v for k,v in VI_TO_EN_COLOR.items()}
def resolve_color(text):
    if not isinstance(text,str) or not text.strip():return None
    normalized=_normalize(text).strip(' .')
    normalized=re.sub(r'^(?:mau|color|colour)\s*:?\s+','',normalized)
    if normalized in LOOKUP:return LOOKUP[normalized]
    # English aliases, only for an unambiguous single color.
    colors,status,_,review=normalize_color(normalized)
    if status=='recognized_single' and not review:return colors[0]
    return None