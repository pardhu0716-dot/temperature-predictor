#!/usr/bin/env python3
"""Geometry-only Stage B generator. Standard library; internal units: integer µm."""
import argparse
from dataclasses import dataclass, replace, asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'dataset/Sample_dataset/2D_floorplans'

@dataclass(frozen=True)
class Block:
    name: str
    w: int
    h: int
    x: int
    y: int


def read_flp(path):
    result = []
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f'{path}: expected five fields')
        values = [float(v) * 1_000_000 for v in fields[1:]]
        if any(not math.isfinite(v) or abs(v-round(v)) > 1e-5 for v in values):
            raise ValueError('Geometry must be finite and on the 1 µm grid')
        result.append(Block(fields[0], *(round(v) for v in values)))
    return result


def write_flp(path, blocks):
    Path(path).write_text('# name width height left-x bottom-y (metres)\n' + ''.join(
        f'{b.name} {b.w/1e6:.6f} {b.h/1e6:.6f} {b.x/1e6:.6f} {b.y/1e6:.6f}\n' for b in blocks))


def bounds(blocks):
    return max(b.x+b.w for b in blocks), max(b.y+b.h for b in blocks)


def overlap(a, b, gap=0):
    return (a.x < b.x+b.w+gap and b.x < a.x+a.w+gap and
            a.y < b.y+b.h+gap and b.y < a.y+a.h+gap)


def validate(blocks, width, height, complete=True, cores=True, gap=0):
    if width <= 0 or height <= 0 or not blocks:
        raise ValueError('Empty/nonpositive die')
    if len({b.name for b in blocks}) != len(blocks):
        raise ValueError('Duplicate block names')
    for b in blocks:
        if not b.name or any(c.isspace() for c in b.name) or '#' in b.name:
            raise ValueError('Invalid block name')
        if min(b.w,b.h) <= 0 or min(b.x,b.y) < 0 or b.x+b.w > width or b.y+b.h > height:
            raise ValueError('Block outside die or nonpositive size')
    for a,b in itertools.combinations(blocks,2):
        if overlap(a,b):
            raise ValueError(f'Overlap: {a.name}, {b.name}')
    active = [b for b in blocks if b.name.startswith('Core_')]
    if cores:
        if len(active) != 8 or sorted((b.w,b.h) for b in active) != [(2000,2000)]*6+[(4000,4000)]*2:
            raise ValueError('Expected exactly 2 Big + 6 Little at original dimensions')
        if any(overlap(a,b,gap) for a,b in itertools.combinations(active,2)):
            raise ValueError('Minimum core spacing violated')
        if any(not b.name.startswith(('Core_', 'FILL_')) for b in blocks):
            raise ValueError('Unknown silicon block')
    if complete and sum(b.w*b.h for b in blocks) != width*height:
        raise ValueError('Incomplete die coverage')


def fill_whitespace(cores, width, height):
    """Exact complement: horizontal free runs, vertically merged when identical."""
    validate(cores,width,height,complete=False)
    xs=sorted({0,width,*[v for b in cores for v in (b.x,b.x+b.w)]})
    ys=sorted({0,height,*[v for b in cores for v in (b.y,b.y+b.h)]})
    fills=[]; previous={}
    for y,y1 in zip(ys,ys[1:]):
        runs=[]; start=None
        for x,x1 in zip(xs,xs[1:]):
            occupied=any(b.x <= x and x1 <= b.x+b.w and b.y <= y and y1 <= b.y+b.h for b in cores)
            if not occupied and start is None: start=x
            if occupied and start is not None: runs.append((start,x)); start=None
        if start is not None: runs.append((start,width))
        current={}
        for x,x1 in runs:
            key=(x,x1)
            if key in previous:
                i=previous[key]; fills[i]=replace(fills[i],h=fills[i].h+y1-y)
            else:
                i=len(fills); fills.append(Block(f'FILL_{i}_L0',x1-x,y1-y,x,y))
            current[key]=i
        previous=current
    return cores+fills


def descriptor(cores, width, height):
    """Name/permutation/translation/reflection/rotation invariant distance summary.

    Conservative heuristic, not a proof of geometric equivalence. Exact duplicates
    necessarily have zero distance; some different layouts may also be rejected.
    """
    parts=[]
    for types in [(2000,2000),(2000,4000),(4000,4000)]:
        parts.extend(sorted(math.hypot(a.x+a.w/2-b.x-b.w/2,a.y+a.h/2-b.y-b.h/2)
            for a,b in itertools.combinations(cores,2) if tuple(sorted((a.w,b.w))) == types))
    return sorted((width,height))+parts


def distance(a,b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/len(a))


def catalog(source):
    entries=[]
    for p in sorted(Path(source).glob('Style_*/*/*_layer_0.flp')):
        blocks=read_flp(p); w,h=bounds(blocks); validate(blocks,w,h)
        tim=p.with_name(p.name.replace('_layer_0','_tim'))
        validate(read_flp(tim),w,h,cores=False)
        lcf=p.with_name(p.name.replace('_layer_0.flp','.lcf'))
        lines=[s.strip() for s in lcf.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
        expected=['0','Y','Y','1.75e6','0.01','0.00015',p.name,'1','Y','N','4e6','0.25','2.0e-05',tim.name]
        if lines != expected: raise ValueError(f'Unexpected layer assumptions: {lcf}')
        cores=[b for b in blocks if b.name.startswith('Core_')]
        entries.append(dict(path=p,style=p.parents[1].name,cores=cores,width=w,height=h,
                            fills=len(blocks)-8,whitespace=1-56_000_000/(w*h),descriptor=descriptor(cores,w,h)))
    if len(entries)!=23: raise ValueError(f'Expected 23 source layouts, found {len(entries)}')
    return entries


def candidate(template, rng, args):
    cores=template['cores']
    # Expand coordinate ranks (rather than scaling core dimensions).
    xvals=sorted({b.x for b in cores}); yvals=sorted({b.y for b in cores})
    dx=rng.randint(0,args.spacing_um); dy=rng.randint(0,args.spacing_um)
    cores=[replace(b,x=b.x+xvals.index(b.x)*dx,y=b.y+yvals.index(b.y)*dy) for b in cores]
    w,h=bounds(cores)
    w=max(w,template['width'])+rng.randint(0,args.margin_um)
    h=max(h,template['height'])+rng.randint(0,args.margin_um)
    if args.die_um: w,h=args.die_um
    for _ in range(80):
        i=rng.randrange(8); b=cores[i]
        moved=replace(b,x=b.x+rng.randint(-args.jitter_um,args.jitter_um),y=b.y+rng.randint(-args.jitter_um,args.jitter_um))
        trial=cores[:];trial[i]=moved
        try: validate(trial,w,h,complete=False,gap=args.min_gap_um)
        except ValueError: continue
        cores=trial
    if args.positions_json:
        positions=json.loads(args.positions_json.read_text())
        if set(positions)!={b.name for b in cores}: raise ValueError('Positions JSON must list all eight core names')
        if any(len(v)!=2 or any(type(c) is not int for c in v) for v in positions.values()):
            raise ValueError('Positions must be integer [x,y] in µm')
        cores=[replace(b,x=positions[b.name][0],y=positions[b.name][1]) for b in cores]
    validate(cores,w,h,complete=False,gap=args.min_gap_um)
    if max(w,h)>=30000: raise ValueError('Die must be smaller than reference 30 mm spreader')
    return cores,w,h


def generate(args):
    entries=catalog(args.source)
    out=args.output.resolve(); source=args.source.resolve()
    # All writes are confined to the new Stage B tree, even for explicit output paths.
    stage_b=Path(__file__).resolve().parent
    if not out.is_relative_to(stage_b) or out==stage_b or out.is_relative_to(source):
        raise ValueError('Output must be a NEW subdirectory of stage_b')
    if out.exists(): raise FileExistsError(f'Refusing to overwrite {out}')
    rng=random.Random(args.seed); accepted=[]; seen=[e['descriptor'] for e in entries]
    for i in range(args.count):
        style=args.styles[i%len(args.styles)]
        for attempt in range(2000):
            template=rng.choice([e for e in entries if e['style']==style])
            try: cores,w,h=candidate(template,rng,args)
            except ValueError:
                if args.positions_json: raise
                continue
            desc=descriptor(cores,w,h); nearest=min(distance(desc,d) for d in seen)
            if nearest < args.min_distance_um: continue
            blocks=fill_whitespace(cores,w,h);validate(blocks,w,h,gap=args.min_gap_um)
            accepted.append((template,blocks,w,h,nearest,attempt+1));seen.append(desc);break
        else: raise ValueError('Could not satisfy geometry/diversity constraints; no output written')
    out.mkdir(parents=True)
    report=[]
    for i,(t,blocks,w,h,nearest,attempts) in enumerate(accepted,1):
        name=f'b_fp_{i:03d}'; folder=out/name;folder.mkdir()
        silicon=folder/f'{name}_layer_0.flp';write_flp(silicon,blocks)
        write_flp(folder/f'{name}_tim.flp',[Block('TIM_Top',w,h,0,0)])
        (folder/f'{name}.lcf').write_text(f'# Silicon: original material assumptions\n0\nY\nY\n1.75e6\n0.01\n0.00015\n{name}_layer_0.flp\n\n# TIM\n1\nY\nN\n4e6\n0.25\n2.0e-05\n{name}_tim.flp\n')
        # No simulation: zero-power format fixture, explicitly not training data.
        (folder/f'{name}_zero.ptrace').write_text(' '.join(b.name for b in blocks)+'\n'+' '.join('0' for b in blocks)+'\n')
        reread=read_flp(silicon);validate(reread,w,h,gap=args.min_gap_um)
        metadata=dict(id=name,style_family=t['style'],source_template=str(t['path'].relative_to(args.source)),
            source_sha256=hashlib.sha256(t['path'].read_bytes()).hexdigest(),seed=args.seed,
            die_um=[w,h],whitespace_fraction=1-56_000_000/(w*h),fill_blocks=len(blocks)-8,
            nearest_descriptor_rms_um=nearest,attempts=attempts,blocks_um=[asdict(b) for b in blocks],
            validation='PASS: bounds, non-overlap, full coverage, core dimensions/count, file roundtrip',
            hotspot_runtime='NOT RUN',parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');report.append(metadata)
    (out/'manifest.json').write_text(json.dumps(dict(source_count=len(entries),floorplans=report),indent=2)+'\n')
    (out/'source_inventory.json').write_text(json.dumps([{k:str(v) if isinstance(v,Path) else v for k,v in e.items() if k not in ('cores','descriptor')} for e in entries],indent=2)+'\n')
    print(json.dumps([dict(id=m['id'],style=m['style_family'],die_um=m['die_um'],fills=m['fill_blocks'],nearest_um=round(m['nearest_descriptor_rms_um'],2)) for m in report],indent=2))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'pilot')
    p.add_argument('--count',type=int,default=5)
    p.add_argument('--seed',type=int,default=20261107)
    p.add_argument('--styles',nargs='+',choices=[f'Style_{i}' for i in range(1,5)],default=[f'Style_{i}' for i in range(1,5)])
    p.add_argument('--die-um',nargs=2,type=int,metavar=('WIDTH','HEIGHT'))
    p.add_argument('--spacing-um',type=int,default=400,help='maximum added gap per coordinate rank')
    p.add_argument('--margin-um',type=int,default=800,help='maximum extra die width/height')
    p.add_argument('--jitter-um',type=int,default=600)
    p.add_argument('--min-gap-um',type=int,default=0)
    p.add_argument('--min-distance-um',type=int,default=150)
    p.add_argument('--positions-json',type=Path,help='all core names mapped to [x,y] in µm; requires count=1')
    return p

if __name__=='__main__':
    p=parser();args=p.parse_args()
    if not 1<=args.count<=5: p.error('Pilot safety limit: count must be 1–5')
    if min(args.spacing_um,args.margin_um,args.jitter_um,args.min_gap_um)<0 or args.min_distance_um<=0: p.error('Invalid spacing/margin/jitter/distance')
    if args.positions_json and args.count!=1: p.error('Explicit positions require count=1')
    generate(args)
