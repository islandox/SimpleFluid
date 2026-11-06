"""Pure checks for pitzDaily evidence; no physical limits are inferred."""
import csv
import math
from pathlib import Path
import re

FIELDS = ('x','y','z','ux','uy','uz','pressure','k','epsilon','nu_t')


def cell_quality(paths, require_physical_fields=False):
    minimum,maximum,count={},{},0
    for path in paths:
        with Path(path).open(newline='',encoding='utf-8') as stream:
            reader=csv.DictReader(stream)
            names=reader.fieldnames
            required=set(FIELDS if require_physical_fields else ('x','y','ux','uy'))
            if not names or len(names)!=len(set(names)) or not required.issubset(names):
                raise ValueError(f'{path}: incomplete or duplicate cell-field columns')
            numeric=[name for name in FIELDS if name in names]
            for row in reader:
                if None in row:
                    raise ValueError(f'{path}: extra cell-field values')
                for name in numeric:
                    try:
                        value=float(row[name])
                    except (ValueError,TypeError) as error:
                        raise ValueError(f'{path}: invalid {name}') from error
                    if not math.isfinite(value):
                        raise ValueError(f'{path}: nonfinite {name}')
                    if name in ('k','epsilon') and value<=0:
                        raise ValueError(f'{path}: {name} must be positive')
                    if name=='nu_t' and value<0:
                        raise ValueError(f'{path}: nu_t must be nonnegative')
                    minimum[name]=min(minimum.get(name,value),value)
                    maximum[name]=max(maximum.get(name,value),value)
                count+=1
    if not count:
        raise ValueError('cell fields contain no rows')
    return {'cells':count,'minimum':minimum,'maximum':maximum}


def _block(text,key):
    pattern=r'(?<!\w)"?'+re.escape(key)+r'"?\s*\{'
    matches=list(re.finditer(pattern,text))
    if len(matches)!=1:
        raise ValueError(f'expected exactly one {key} sampling dictionary')
    start=matches[0].end()
    depth=1
    for index in range(start,len(text)):
        if text[index]=='{': depth+=1
        elif text[index]=='}': depth-=1
        if depth==0:
            return text[start:index]
    raise ValueError(f'unclosed {key} sampling dictionary')


def validate_profile_control(text,stations):
    """Validate the fixture's declared sample locations, not only file labels."""
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',text,flags=re.S)
    body=_block(text,'profiles')
    sets=list(re.finditer(r'\bsets\s*\(',body))
    if len(sets)!=1:
        raise ValueError('expected one explicit profile set list')
    header=body[:sets[0].start()]
    for key,expected in [('type','sets'),('setFormat','raw'),('interpolationScheme','cellPoint')]:
        values=re.findall(r'\b'+key+r'\s+(\w+)\s*;',header)
        if values!=[expected]:
            raise ValueError(f'profile {key} must be {expected}')
    if re.findall(r'\bfields\s*\(([^()]*)\)\s*;',header)!=['U']:
        # Whitespace inside the list is insignificant.
        values=re.findall(r'\bfields\s*\(([^()]*)\)\s*;',header)
        if len(values)!=1 or values[0].split()!=['U']:
            raise ValueError('profile fields must be exactly U')
    for name,x in stations.items():
        station=_block(body[sets[0].end():],name)
        for key,expected in [('type','uniform'),('axis','y'),('nPoints','129')]:
            if re.findall(r'\b'+key+r'\s+(\w+)\s*;',station)!=[expected]:
                raise ValueError(f'{name}: incorrect {key}')
        for key,y in [('start',-.0253),('end',.0253)]:
            vectors=re.findall(r'\b'+key+r'\s*\(([^()]*)\)\s*;',station)
            try:
                actual=tuple(float(v) for v in vectors[0].split()) if len(vectors)==1 else None
            except ValueError as error:
                raise ValueError(f'{name}: invalid {key} coordinate') from error
            if actual!=(x,y,0.):
                raise ValueError(f'{name}: {key} must match the declared physical station and span')
    return True
