"""Explicit CSV adaptation and strict geometry/label validation; no label generation."""
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import numpy as np

CORE_IDS = tuple(f'Core_{i}_L0' for i in range(8))
FIELDS = ('width', 'height', 'x', 'y', 'power', 'temperature')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def number(value, context):
    try: result = float(value)
    except (ValueError, TypeError) as exc: raise ValueError(f'{context}: expected numeric value') from exc
    if not np.isfinite(result): raise ValueError(f'{context}: nonfinite value')
    return result


@dataclass
class Dataset:
    floorplan_ids: np.ndarray
    sample_ids: np.ndarray
    geometry_mm: np.ndarray  # n,8,4: width,height,x,y, canonical core ID order
    powers_w: np.ndarray
    temperatures_c: np.ndarray
    die_mm: np.ndarray
    source_sha256: str
    schema: dict


def schema_template():
    """EXAMPLE names only: user/Antigravity must map the actual local CSV header."""
    return dict(version=1, floorplan_column='floorplan_id', sample_column='sample_id',
                expected_columns=58, geometry_unit='mm', temperature_unit='C', power_unit='W',
                label_source='HotSpot', ambient_c=45.0, max_temperature_c=500.0,
                simulation_contract=None,
                cores=[dict(id_column=f'core_{i}_id', **{field:f'core_{i}_{field}' for field in FIELDS})
                       for i,name in enumerate(CORE_IDS)])


def validate_schema(schema):
    if schema.get('version') != 1: raise ValueError('Unsupported schema version')
    if schema.get('geometry_unit') not in ('m','mm','um'): raise ValueError('Declare geometry_unit: m, mm or um')
    if schema.get('temperature_unit') not in ('C','K'): raise ValueError('Declare temperature_unit: C or K')
    if schema.get('power_unit') != 'W': raise ValueError('Only explicitly declared Watts supported')
    if schema.get('label_source') != 'HotSpot': raise ValueError('Labels must be declared original HotSpot outputs; Stage A predictions are not ground truth')
    if len(schema.get('cores',[])) != 8: raise ValueError('Map exactly eight cores')
    columns=[schema['floorplan_column'],schema['sample_column']]
    fixed=[]
    for slot in schema['cores']:
        if ('core_id' in slot) == ('id_column' in slot): raise ValueError('Each core mapping needs exactly one of core_id or id_column')
        if 'core_id' in slot: fixed.append(slot['core_id'])
        else: columns.append(slot['id_column'])
        for field in FIELDS:
            if not isinstance(slot.get(field),str) or not slot[field]: raise ValueError(f'Map {field} column explicitly')
            columns.append(slot[field])
    if fixed and (len(set(fixed))!=len(fixed) or any(n not in CORE_IDS for n in fixed)):
        raise ValueError('Invalid/duplicate fixed core identifiers')
    if len(set(columns)) != len(columns): raise ValueError('A CSV column cannot serve multiple roles')
    ambient=number(schema.get('ambient_c'),'ambient_c')
    ceiling=number(schema.get('max_temperature_c'),'max_temperature_c')
    if ceiling<=ambient: raise ValueError('Temperature validation ceiling must exceed ambient')
    return columns


def load_geometry_manifest(path):
    """Reuse committed floorplans for authoritative geometry; no simulator imports."""
    manifest=read_json(path);root=Path(path).resolve().parent;layouts={}
    for f in manifest['floorplans']:
        fid=f['id']
        if fid in layouts: raise ValueError('Duplicate manifest floorplan')
        flp=(root/f['silicon']).resolve()
        if not flp.is_relative_to(root): raise ValueError('Unsafe manifest path')
        expected=f['input_sha256'][f['silicon']]
        if digest(flp)!=expected: raise ValueError(f'Geometry hash changed: {fid}')
        blocks=[];cores={}
        for line in flp.read_text().splitlines():
            tokens=line.split('#',1)[0].split()
            if not tokens: continue
            if len(tokens)!=5: raise ValueError('Invalid FLP row')
            name=tokens[0];v=np.asarray([number(x,name) for x in tokens[1:]])*1000
            w,h,x,y=v
            if w<=0 or h<=0 or x<0 or y<0: raise ValueError('Invalid FLP geometry')
            if any(name==b[0] for b in blocks): raise ValueError('Duplicate FLP block')
            blocks.append((name,v))
            if name.startswith('Core_'):cores[name]=v
            elif not name.startswith('FILL_'): raise ValueError('Unknown silicon block')
        lcf=(root/f['lcf']).resolve()
        if not lcf.is_relative_to(root) or digest(lcf)!=f['input_sha256'][f['lcf']]:raise ValueError('LCF hash changed')
        lines=[line.strip() for line in lcf.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
        if len(lines)!=14 or lines[:6]!=['0','Y','Y','1.75e6','0.01','0.00015'] or lines[7:13]!=['1','Y','N','4e6','0.25','2.0e-05']:
            raise ValueError('Unsupported material/layer assumptions; these features assume the original silicon/TIM stack')
        die=np.asarray(f['die_um'],dtype=float)/1000
        validate_geometry(np.asarray([cores[n] for n in CORE_IDS]) if set(cores)==set(CORE_IDS) else np.zeros((0,4)),die)
        for i,(_,v) in enumerate(blocks):
            w,h,x,y=v
            if x+w>die[0]+1e-8 or y+h>die[1]+1e-8: raise ValueError('FLP block outside die')
            for _,other in blocks[:i]:
                ow,oh,ox,oy=other
                if min(x+w,ox+ow)-max(x,ox)>1e-8 and min(y+h,oy+oh)-max(y,oy)>1e-8:raise ValueError('FLP overlap')
        if not np.isclose(sum(v[0]*v[1] for _,v in blocks),np.prod(die),atol=1e-7,rtol=0):raise ValueError('Incomplete FILL coverage')
        layouts[fid]=(np.asarray([cores[n] for n in CORE_IDS]),die)
    return layouts


def validate_geometry(g,die):
    if g.shape != (8,4) or die.shape != (2,) or not np.isfinite(g).all() or not np.isfinite(die).all():raise ValueError('Expected eight finite core rectangles and die dimensions')
    if np.any(die<=0) or np.any(g[:,:2]<=0) or np.any(g[:,2:]<0):raise ValueError('Nonpositive sizes/negative coordinates')
    big=np.all(np.isclose(g[:,:2],4,atol=1e-6,rtol=0),axis=1)
    little=np.all(np.isclose(g[:,:2],2,atol=1e-6,rtol=0),axis=1)
    if big.sum()!=2 or little.sum()!=6:raise ValueError('Require 2 Big 4x4 mm and 6 Little 2x2 mm')
    if np.any(g[:,2:]+g[:,:2]>die+1e-6):raise ValueError('Core outside die')
    for i in range(8):
        for j in range(i):
            a,b=g[i],g[j]
            if min(a[2]+a[0],b[2]+b[0])-max(a[2],b[2])>1e-8 and min(a[3]+a[1],b[3]+b[1])-max(a[3],b[3])>1e-8:raise ValueError('Core overlap')
    return big


def load_power_inputs(geometry_manifest):
    root=Path(geometry_manifest).resolve().parent;manifest=read_json(geometry_manifest)
    wp=(root/manifest['workload_manifest']).resolve()
    if not wp.is_relative_to(root) or digest(wp)!=manifest['workload_manifest_sha256']:raise ValueError('Workload manifest hash mismatch')
    powers={}
    for work in read_json(wp)['workloads']:
        path=(root/work['ptrace']).resolve()
        if not path.is_relative_to(root) or digest(path)!=work['sha256']:raise ValueError('Original power input changed')
        rows=[s.split() for s in path.read_text().splitlines() if s.strip()]
        if len(rows)!=2 or len(rows[0])!=len(rows[1]) or len(set(rows[0]))!=len(rows[0]):raise ValueError('Invalid original ptrace')
        values={k:number(v,'ptrace power') for k,v in zip(rows[0],rows[1])}
        if any(v!=0 for k,v in values.items() if k.startswith('FILL_')):raise ValueError('Nonzero passive FILL')
        key=(work['floorplan_id'],work['sample_id'])
        if key in powers:raise ValueError('Duplicate original workload')
        powers[key]=np.asarray([values[n] for n in CORE_IDS])
    return powers


def load_csv(path,schema,geometry_manifest, *, expected_floorplans=77, samples_per_floorplan=20, external=False):
    required=validate_schema(schema);layouts=load_geometry_manifest(geometry_manifest)
    if not external and set(layouts)!={f'b_fp_{i:03d}' for i in range(6,83)}:raise ValueError('Training manifest must contain only b_fp_006–b_fp_082; original Stage A layouts excluded')
    original_powers=load_power_inputs(geometry_manifest) if not external else None
    geometry=[];powers=[];targets=[];dies=[];ids=[];samples=[];keys=set();vectors=set();counts={}
    factor={'m':1000.,'mm':1.,'um':.001}[schema['geometry_unit']]
    with Path(path).open(newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        header=reader.fieldnames or []
        if len(header)!=len(set(header)):raise ValueError('Duplicate CSV headers')
        if schema.get('expected_columns') is not None and len(header)!=schema['expected_columns']:raise ValueError(f'Expected {schema["expected_columns"]} columns, got {len(header)}')
        if set(required)-set(header):raise ValueError(f'Missing mapped columns: {sorted(set(required)-set(header))}. Inspect header and edit mapping; no schema is guessed.')
        for rownum,row in enumerate(reader,2):
            if None in row or any(v is None or not str(v).strip() for v in row.values()):raise ValueError(f'Row {rownum}: missing value or ragged row')
            fid=row[schema['floorplan_column']].strip();sid=row[schema['sample_column']].strip()
            if fid not in layouts:raise ValueError(f'Unknown/ineligible floorplan {fid}')
            if not external and not re.fullmatch(r'sample_\d{3}',sid):raise ValueError('Sample IDs must match the original workload IDs sample_001 etc.; adapt the CSV explicitly if necessary')
            if (fid,sid) in keys:raise ValueError(f'Duplicate floorplan/sample {fid}/{sid}')
            keys.add((fid,sid));mapped={}
            for slot in schema['cores']:
                name=slot.get('core_id') if 'core_id' in slot else row[slot['id_column']].strip()
                if name in mapped or name not in CORE_IDS:raise ValueError('Invalid/duplicate row core ID')
                mapped[name]=[number(row[slot[k]],f'row {rownum}/{name}/{k}') for k in FIELDS]
            if set(mapped)!=set(CORE_IDS):raise ValueError('Missing core identifiers')
            values=np.asarray([mapped[n] for n in CORE_IDS]);g=values[:,:4]*factor;p=values[:,4];y=values[:,5]
            if schema['temperature_unit']=='K':y=y-273.15
            original,die=layouts[fid];big=validate_geometry(g,die)
            if not np.allclose(g,original,atol=1e-5,rtol=0):raise ValueError(f'CSV geometry/core-ID mapping differs from committed FLP: {fid}')
            low=np.where(big,10.,5.);high=np.where(big,20.,10.)
            if np.any(p<low-1e-8) or np.any(p>high+1e-8):raise ValueError('Powers outside original Big/Little ranges')
            if not external and ((fid,sid) not in original_powers or not np.allclose(p,original_powers[(fid,sid)],atol=1e-6,rtol=0)):raise ValueError('CSV powers/sample ID differ from original workload manifest')
            if np.any(y<float(schema['ambient_c'])-.05) or np.any(y>float(schema['max_temperature_c'])):raise ValueError('Temperature outside declared validation limits; verify units and labels')
            signature=(fid,*p.tolist())
            if signature in vectors:raise ValueError('Duplicate power configuration within a floorplan')
            vectors.add(signature);counts[fid]=counts.get(fid,0)+1
            geometry.append(g);powers.append(p);targets.append(y);dies.append(die);ids.append(fid);samples.append(sid)
    if not ids:raise ValueError('Empty CSV')
    if expected_floorplans is not None and len(counts)!=expected_floorplans:raise ValueError(f'Expected {expected_floorplans} floorplans, found {len(counts)}')
    if set(counts)!=set(layouts):raise ValueError('CSV and geometry manifest floorplan sets differ')
    if samples_per_floorplan is not None:
        if any(n!=samples_per_floorplan for n in counts.values()):raise ValueError('Unexpected configurations per floorplan')
        if not external and any({s for f,s in keys if f==fid}!={f'sample_{i:03d}' for i in range(1,samples_per_floorplan+1)} for fid in counts):raise ValueError('Incomplete sample IDs')
    schema=dict(schema,observed_columns=header)
    return Dataset(np.asarray(ids),np.asarray(samples),np.asarray(geometry),np.asarray(powers),np.asarray(targets),np.asarray(dies),digest(path),schema)
