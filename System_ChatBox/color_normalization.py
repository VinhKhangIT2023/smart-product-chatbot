"""Conservative, deterministic color taxonomy. No inference from product titles."""
import re
import unicodedata

VERSION = 'color-v3.1'
CANONICAL = ('White', 'Black', 'Gray', 'Silver', 'Blue', 'Red', 'Brown',
             'Clear', 'Green', 'Pink', 'Gold', 'Beige', 'Yellow', 'Purple',
             'Orange', 'Ivory', 'Cream', 'Bronze', 'Teal', 'Turquoise',
             'Copper', 'Rose Gold', 'Multicolor')

def key(value):
    value = unicodedata.normalize('NFKC', value).casefold().strip()
    return re.sub(r'\s+', ' ', value.replace('–', '-').replace('—', '-'))

ALIASES = {key(c): c for c in CANONICAL}
GROUPS = {
    'Gray': ['grey', 'dark gray', 'dark grey', 'light gray', 'light grey', 'charcoal gray', 'charcoal grey'],
    'White': ['pure white', 'bright white', 'snow white'],
    'Blue': ['navy', 'navy blue', 'light blue', 'dark blue', 'royal blue', 'sky blue', 'cobalt blue', 'baby blue', 'aqua blue'],
    'Green': ['light green', 'dark green', 'sage green', 'mint green', 'olive green', 'forest green', 'emerald green'],
    'Brown': ['dark brown', 'light brown', 'chocolate brown', 'rustic brown'],
    'Pink': ['light pink', 'dark pink', 'hot pink', 'blush pink', 'rose pink'],
    'Red': ['dark red', 'light red', 'wine red', 'rose red'],
    'Purple': ['light purple', 'dark purple', 'violet', 'lavender purple'],
    'Orange': ['burnt orange', 'dark orange', 'light orange'],
    'Yellow': ['mustard yellow', 'light yellow', 'dark yellow', 'lemon yellow'],
    'Clear': ['transparent', 'colourless', 'colorless'],
    'Gold': ['golden'],
    'Silver': ['sliver'],
    'Multicolor': ['multi', 'multi color', 'multi-color', 'multi colored', 'multi-colored',
                  'multicolored', 'multicolour', 'multicoloured', 'multi colour', 'multi-colour', 'colorful', 'rainbow'],
    'Cream': ['off white', 'off-white'],
}
for canonical, aliases in GROUPS.items():
    ALIASES.update({key(a): canonical for a in aliases})
# Only unambiguous finish + a recognized color. Materials are never colors.
for prefix in ['matte', 'glossy', 'gloss', 'satin', 'metallic']:
    for c in CANONICAL:
        if c not in ['Multicolor','Clear']:
            ALIASES[prefix+' '+key(c)] = c

MATERIALS = {'stainless steel','stainless','steel','glass','wood','natural wood','bamboo',
             'aluminum','aluminium','plastic','silicone','ceramic','porcelain','metal','brass',
             'nickel','brushed nickel','satin nickel','chrome'}
NON_COLORS = {'round','grain mill','spiderman','rosewood 3 in 1 grill brush',
              'original version','unframed','unframed version','unframed paper',
              'premium unframed version','poster'}
NULLS = {'','none','null','nan','n/a','unknown','not available'}
COUNT = re.compile(r'\d+\s*(?:-\s*)?(?:pack|packs|pcs|pieces|piece|count|ct)',re.I)

def normalize_color(value):
    raw = '' if value is None else str(value)
    k = key(raw)
    if k in NULLS: return [], 'missing', 'empty_or_null', False
    if k in ALIASES:
        return [ALIASES[k]], 'recognized_single', 'exact_alias', False
    if k in MATERIALS or k in NON_COLORS or COUNT.fullmatch(k):
        return [], 'non_color', 'known_non_color', False
    # Split only explicit list separators; do not search color words inside prose.
    parts = [s.strip() for s in re.split(r'\s*(?:/|,|;|&|\+|\band\b)\s*', k) if s.strip()]
    if len(parts) == 1:
        return [], 'needs_review', 'unmapped_whole_value', True
    colors, ignored, unknown = [], [], []
    for part in parts:
        if part in ALIASES:
            colors.append(ALIASES[part])
        elif part in MATERIALS or COUNT.fullmatch(part):
            ignored.append(part)
        else:
            unknown.append(part)
    if unknown:
        return [], 'needs_review', 'unmapped_component', True
    colors = sorted(set(colors), key=CANONICAL.index)
    if not colors:
        return [], 'non_color', 'only_material_or_quantity', False
    if ignored:
        return colors, 'recognized_with_metadata', 'explicit_colors_only_metadata_ignored', True
    return colors, 'recognized_multi' if len(colors)>1 else 'recognized_single', 'explicit_color_list', False

def color_fields(raw):
    import json
    colors, status, rule, review = normalize_color(raw)
    return {'color': ' | '.join(colors), 'color_raw': raw,
            'colors_json': json.dumps(colors,ensure_ascii=False),
            'color_status':status,'color_rule':rule,'color_needs_review':int(review)}