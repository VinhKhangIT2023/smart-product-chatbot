"""Conservative two-dimensional extraction, size-v4.1. Standard library only."""
import re

VERSION='size-v4.1'
FACTORS={'cm':1.0,'mm':0.1,'m':100.0,'in':2.54,'ft':30.48}
N=r'\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?'
U=r'(?:inches|inch|feet|foot|centimeters?|centimetres?|millimeters?|millimetres?|meters?|metres?|cm|mm|ft|in|m|[\"\'])'
AX=rf'(?:{N})\s*(?:{U})?(?:\s*(?:{N})\s*(?:inches|inch|in|"))?\s*(?:[LWDH](?=\s|[x×*]|$))?'
PAIR=re.compile(rf'(?<![\w.])(?P<a>{AX})\s*[x×*]\s*(?P<b>{AX})(?:\s*[x×*]\s*(?P<c>{AX}))?(?![\w.])',re.I)
MEASURE=re.compile(rf'(?P<n>{N})\s*(?P<u>{U})?',re.I)
DIAM=re.compile(rf'\bdiameter\s*[:=]?\s*(?P<a>(?:{N})\s*{U})(?!\w)',re.I)
KEYWORD=re.compile(r'\b(?:size|dimensions?|measures?|measurements?)\b',re.I)
EXCLUDE=re.compile(r'\b(?:package|packaging|shipping|carton|box size|fits?|compatible|pillow\s*insert)\b',re.I)

def unit(u):
    if not u:return None
    u=u.lower()
    if u in ['"','in','inch','inches']:return 'in'
    if u in ["'",'ft','feet','foot']:return 'ft'
    if u=='mm' or u.startswith('milli'):return 'mm'
    if u=='cm' or u.startswith('centi'):return 'cm'
    if u=='m' or u.startswith('met'):return 'm'
    raise ValueError(u)

def number(s):
    return sum(float(p.split('/')[0])/float(p.split('/')[1]) if '/' in p else float(p) for p in s.split())

def axis(s):
    pieces=list(MEASURE.finditer(s))
    if len(pieces)==1:
        m=pieces[0];return number(m['n']),unit(m['u'])
    if len(pieces)==2 and unit(pieces[0]['u'])=='ft' and unit(pieces[1]['u'])=='in':
        return number(pieces[0]['n'])*30.48+number(pieces[1]['n'])*2.54,'cm'
    raise ValueError('unsupported axis')

def deviation(a,b):
    return max(abs(x-y)/max(abs(x),0.000001) for x,y in zip(a,b))

def parse_dimensions(text,source='size_raw'):
    text=(text or '').replace('′',"'").replace('″','"').replace('“','"').replace('”','"').replace('’',"'")
    candidates=[];rejected=[]
    for m in list(PAIR.finditer(text))+list(DIAM.finditer(text)):
        prefix=text[max(0,m.start()-65):m.start()]
        # Use only the same short sentence/clause before the measurement.
        prefix=re.split(r'[.!?;\n]',prefix)[-1]
        if EXCLUDE.search(prefix):rejected.append('excluded_context');continue
        if re.search(r'[-–]\s*$',prefix):rejected.append('range_or_negative');continue
        if re.match(r'\s*[x×*]\s*\d',text[m.end():]):rejected.append('more_than_three_axes');continue
        if re.search(r'\d\s*[-–]\s*$',prefix):rejected.append('range');continue
        if source=='features' and not KEYWORD.search(prefix):rejected.append('missing_keyword');continue
        try:
            values=[axis(m['a'])] if m.re is DIAM else [axis(m['a']),axis(m['b'])]+([axis(m['c'])] if m['c'] else [])
            if not any(u for _,u in values):rejected.append('missing_unit');continue
            # Missing units are permitted only before the last axis, inheriting its unit.
            if values[-1][1] is None:rejected.append('ambiguous_unit');continue
            last_unit=values[-1][1]
            dims=[n*FACTORS[u or last_unit] for n,u in values]
            if any(n<=0 or n>100000 for n in dims):rejected.append('invalid_or_extreme');continue
            dims=dims[:2] if len(dims)>1 else dims*2
            if any(round(v,2)<=0 for v in dims):rejected.append('rounds_to_zero');continue
            dims=sorted(dims,reverse=True)
            candidates.append({'dims':dims,'evidence':m.group(),'shape':'diameter' if m.re is DIAM else 'first_two_axes','third_axis_ignored':len(values)==3})
        except (ValueError,ZeroDivisionError):rejected.append('unsupported_number')
    if not candidates:return {'dims':None,'reason':','.join(sorted(set(rejected))) or 'no_explicit_dimensions','candidates':[]}
    if any(deviation(candidates[0]['dims'],c['dims'])>0.05 for c in candidates[1:]):
        return {'dims':None,'reason':'multiple_inconsistent_measurements','candidates':candidates}
    return {**candidates[0],'reason':'explicit_pair_or_diameter','candidates':candidates}

def extract_size(row):
    sources=['size_raw','title','features','product_dimensions_raw']
    parsed={s:parse_dimensions(row.get(s,''),s) for s in sources}
    chosen=next((s for s in sources if parsed[s]['dims']),None)
    ambiguous=any(x['reason']=='multiple_inconsistent_measurements' for x in parsed.values())
    text_present=any(row.get(s,'').strip() for s in ['size_raw','product_dimensions_raw'])
    if not chosen:
        status='conflict' if ambiguous else ('non_dimension' if text_present else 'missing')
        return {'size_dim1_cm':'','size_dim2_cm':'','size_source':'','size_confidence':'','size_status':status,
                'size_needs_review':int(ambiguous or text_present),'size_evidence':'','size_rule':VERSION+':no_selected_measurement','size_basis':''},parsed
    dims=parsed[chosen]['dims']
    conflict=ambiguous or any(deviation(dims,v['dims'])>0.15 for s,v in parsed.items() if s!=chosen and v['dims'])
    conf={'size_raw':'high','title':'high','features':'medium','product_dimensions_raw':'low'}[chosen]
    return {'size_dim1_cm':round(dims[0],2),'size_dim2_cm':round(dims[1],2),'size_source':chosen,
            'size_confidence':conf,'size_status':'conflict' if conflict else 'recognized',
            'size_needs_review':int(conflict or conf=='low'),'size_evidence':parsed[chosen]['evidence'],
            'size_rule':VERSION+(':priority_with_conflict' if conflict else ':priority'),
            'size_basis':parsed[chosen]['shape']},parsed

def parse_size_request(text):
    """Explicit numeric request only. Return sorted cm pair, or None; not a room-fit inference."""
    p=parse_dimensions(text)
    return tuple(round(v,2) for v in p['dims']) if p['dims'] else None