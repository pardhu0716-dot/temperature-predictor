#!/usr/bin/env python3
"""Prepare geometry and power inputs only. Never invokes HotSpot or trains a model."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
from types import SimpleNamespace

from generate_floorplans import (Block, SOURCE, bounds, candidate, catalog, descriptor,
    distance, fill_whitespace, overlap, read_flp, validate, write_flp)
from generate_power_traces import (analyze_floorplan, generate_random_traces,
    generate_reference_trace, write_ptrace, CoreType)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BASE_COMMIT = 'c8cc4e27a332af168b9e539aae80da615735eeca'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def resolve_input(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Manifest paths must be relative strings')
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Missing/unsafe input path: {relative}')
    return path


def read_trace(path, profile, big_range, little_range, reference=False):
    rows = [s.split() for s in Path(path).read_text().splitlines() if s.strip()]
    if len(rows) != 2 or rows[0] != profile.header_order or len(rows[1]) != len(rows[0]):
        raise ValueError(f'Invalid single steady-state ptrace/header: {path}')
    powers = [float(v) for v in rows[1]]
    if any(not math.isfinite(v) or v < 0 for v in powers):
        raise ValueError(f'Nonfinite/negative powers: {path}')
    for block, value in zip(profile.blocks, powers):
        if block.block_type == CoreType.FILL:
            if value != 0: raise ValueError(f'Nonzero FILL: {path}')
        elif reference:
            expected = 20 if block.block_type == CoreType.BIG else 8
            if value != expected: raise ValueError(f'Incorrect reference power: {path}')
        else:
            lo, hi = big_range if block.block_type == CoreType.BIG else little_range
            if not lo <= value <= hi: raise ValueError(f'Power out of range: {path}')
    return dict(zip(rows[0], powers))


def check_layers(lcf, silicon, width, height):
    lines = [s.strip() for s in lcf.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    if len(lines) != 14: raise ValueError(f'Expected two LCF layers: {lcf}')
    if lines[:6] != ['0','Y','Y','1.75e6','0.01','0.00015'] or lines[7:13] != ['1','Y','N','4e6','0.25','2.0e-05']:
        raise ValueError(f'Changed layer/material assumptions: {lcf}')
    layers = [resolve_input(lcf.parent, lines[i]) for i in (6,13)]
    if layers[0] != silicon: raise ValueError('LCF silicon reference mismatch')
    tim = read_flp(layers[1]); validate(tim,width,height,cores=False)
    if tim != [Block('TIM_Top',width,height,0,0)]: raise ValueError('Expected full-die TIM')
    return layers


def prior_descriptors():
    entries = catalog(SOURCE)
    seen = [e['descriptor'] for e in entries]
    pilots = sorted((HERE/'pilot').glob('b_fp_*/*_layer_0.flp'))
    if len(pilots) != 5: raise ValueError('Expected five preserved pilot layouts')
    for p in pilots:
        blocks=read_flp(p); w,h=bounds(blocks);validate(blocks,w,h)
        seen.append(descriptor([b for b in blocks if b.name.startswith('Core_')],w,h))
    return entries,seen


def validate_dataset(manifest_path, check_diversity=True):
    """Validate every geometry, input hash, LCF, and trace; returns resolved jobs."""
    manifest_path=Path(manifest_path).resolve();root=manifest_path.parent
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1: raise ValueError('Unsupported manifest version')
    workload_path=resolve_input(root,manifest['workload_manifest'])
    if sha256(workload_path) != manifest['workload_manifest_sha256']: raise ValueError('Workload manifest hash mismatch')
    workloads=json.loads(workload_path.read_text())
    if workloads.get('schema_version') != 1: raise ValueError('Unsupported workload schema')
    if workloads['configuration_count'] != len(workloads['workloads']): raise ValueError('Workload count mismatch')
    if manifest['new_floorplan_count'] != len(manifest['floorplans']): raise ValueError('Floorplan count mismatch')
    if manifest['expected_simulations'] != len(workloads['workloads']): raise ValueError('Simulation count mismatch')
    _,seen=prior_descriptors() if check_diversity else (None,[])
    floors={};jobs=[];seen_jobs=set();seen_powers={}
    for f in manifest['floorplans']:
        fid=f['id']
        if fid in floors or not fid.startswith('b_fp_') or not fid[5:].isdigit(): raise ValueError('Invalid/duplicate floorplan ID')
        if int(fid[5:]) < 6: raise ValueError('Manifest may not select existing pilots')
        flp=resolve_input(root,f['silicon']);lcf=resolve_input(root,f['lcf'])
        for rel,digest in f['input_sha256'].items():
            if sha256(resolve_input(root,rel)) != digest: raise ValueError(f'Changed input: {rel}')
        blocks=read_flp(flp);w,h=f['die_um'];validate(blocks,w,h)
        layers=check_layers(lcf,flp,w,h)
        if set(f['input_sha256']) != {str(p.relative_to(root)) for p in [lcf,*layers]}:
            raise ValueError('Incomplete input hash inventory')
        cores=[b for b in blocks if b.name.startswith('Core_')]
        desc=descriptor(cores,w,h)
        if check_diversity and min(distance(desc,d) for d in seen) < manifest['parameters']['min_distance_um']:
            raise ValueError(f'Duplicate/near-identical geometry: {fid}')
        seen.append(desc)
        profile=analyze_floorplan(flp)
        if sorted(profile.big_cores) != f['big_cores'] or sorted(profile.little_cores) != f['little_cores']:
            raise ValueError('Core classification mismatch')
        if set(profile.big_cores+profile.little_cores) != {f'Core_{i}_L0' for i in range(8)}:
            raise ValueError('Core ID set mismatch')
        floors[fid]=(f,flp,lcf,layers,profile);seen_powers[fid]=set()
    for work in workloads['workloads']:
        fid=work['floorplan_id'];sid=work['sample_id'];key=(fid,sid)
        if fid not in floors or key in seen_jobs or not sid.startswith('sample_') or not sid[7:].isdigit():
            raise ValueError('Invalid/duplicate workload ID')
        seen_jobs.add(key)
        f,flp,lcf,layers,profile=floors[fid]
        ptrace=resolve_input(root,work['ptrace'])
        if sha256(ptrace)!=work['sha256']:raise ValueError(f'Changed ptrace: {ptrace}')
        reference=sid=='sample_001'
        if work['reference'] != reference:raise ValueError('Reference must be sample_001')
        powers=read_trace(ptrace,profile,workloads['big_range_w'],workloads['little_range_w'],reference)
        signature=tuple(powers[c] for c in sorted(profile.big_cores+profile.little_cores))
        if signature in seen_powers[fid]:raise ValueError(f'Duplicate power configuration: {fid}')
        seen_powers[fid].add(signature)
        jobs.append(dict(floorplan_id=fid,sample_id=sid,flp=flp,lcf=lcf,layers=layers,
                         ptrace=ptrace,profile=profile,powers=powers))
    n=workloads['samples_per_floorplan']
    for fid in floors:
        if {sid for f,sid in seen_jobs if f==fid} != {f'sample_{i:03d}' for i in range(1,n+1)}:
            raise ValueError(f'Incomplete workload set: {fid}')
    return manifest,jobs


def prepare(output, count=77, samples=20, seed=20261108, big_range=(10.,20.), little_range=(5.,10.),
            min_distance_um=150):
    output=Path(output).resolve()
    if output.parent != HERE or output.name in ('pilot','simulation_results'):
        raise ValueError('Output must be a new direct child of stage_b, outside pilot/results')
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}')
    if not 1 <= count <= 1000 or samples < 1 or min_distance_um <= 0:raise ValueError('Invalid count/samples/distance')
    entries,seen=prior_descriptors()
    params=dict(count=count,samples=samples,seed=seed,big_range_w=list(big_range),little_range_w=list(little_range),
                min_distance_um=min_distance_um,spacing_um=800,margin_um=1600,jitter_um=1000)
    geometry_args=SimpleNamespace(spacing_um=800,margin_um=1600,jitter_um=1000,
                                  die_um=None,positions_json=None,min_gap_um=0)
    # Build and validate in a temporary tree, then publish the completed dataset.
    temp=Path(tempfile.mkdtemp(prefix='.preparing-',dir=HERE))
    try:
        floorplans=[];workloads=[]
        for index in range(count):
            fid=f'b_fp_{index+6:03d}';style=f'Style_{index%4+1}'
            geom_seed=seed+index; rng=random.Random(geom_seed)
            for attempt in range(5000):
                template=rng.choice([e for e in entries if e['style']==style])
                try:cores,w,h=candidate(template,rng,geometry_args)
                except ValueError:continue
                desc=descriptor(cores,w,h);nearest=min(distance(desc,d) for d in seen)
                if nearest < min_distance_um:continue
                # IDs and file order vary independently of dimensions and placement.
                numbers=list(range(8));rng.shuffle(numbers)
                cores=[replace(b,name=f'Core_{number}_L0') for b,number in zip(cores,numbers)]
                rng.shuffle(cores)
                blocks=fill_whitespace(cores,w,h);validate(blocks,w,h)
                seen.append(desc);break
            else:raise RuntimeError(f'Cannot find distinct geometry for {fid}')
            folder=temp/'floorplans'/fid;folder.mkdir(parents=True)
            flp=folder/f'{fid}_layer_0.flp';tim=folder/f'{fid}_tim.flp';lcf=folder/f'{fid}.lcf'
            write_flp(flp,blocks);write_flp(tim,[Block('TIM_Top',w,h,0,0)])
            lcf.write_text(f'# Silicon: original assumptions\n0\nY\nY\n1.75e6\n0.01\n0.00015\n{flp.name}\n\n# TIM\n1\nY\nN\n4e6\n0.25\n2.0e-05\n{tim.name}\n')
            profile=analyze_floorplan(flp)
            work_seed=int.from_bytes(hashlib.sha256(f'{seed}:{fid}:powers'.encode()).digest()[:8],'big')
            traces=generate_random_traces(profile,samples,seed=work_seed,big_range=big_range,
                                          little_range=little_range,include_reference=True)
            trace_dir=temp/'workloads'/fid;trace_dir.mkdir(parents=True)
            for trace in traces:
                ptrace=trace_dir/f'{trace.sample_id}.ptrace';write_ptrace(ptrace,trace)
                read_trace(ptrace,profile,big_range,little_range,trace.sample_id=='sample_001')
                workloads.append(dict(floorplan_id=fid,sample_id=trace.sample_id,reference=trace.sample_id=='sample_001',
                                      ptrace=str(ptrace.relative_to(temp)),sha256=sha256(ptrace)))
            item=dict(id=fid,style_family=style,source_template=str(template['path'].relative_to(PROJECT)),
                      source_sha256=sha256(template['path']),geometry_seed=geom_seed,power_seed=work_seed,
                      die_um=[w,h],big_cores=sorted(profile.big_cores),little_cores=sorted(profile.little_cores),
                      fill_count=len(blocks)-8,whitespace_fraction=1-56_000_000/(w*h),
                      nearest_descriptor_rms_um=nearest,attempts=attempt+1,
                      silicon=str(flp.relative_to(temp)),lcf=str(lcf.relative_to(temp)),
                      input_sha256={str(p.relative_to(temp)):sha256(p) for p in (flp,tim,lcf)})
            save_json(folder/'metadata.json',item);floorplans.append(item)
        wp=temp/'workload_manifest.json'
        save_json(wp,dict(schema_version=1,configuration_count=len(workloads),samples_per_floorplan=samples,
                          big_range_w=list(big_range),little_range_w=list(little_range),seed=seed,
                          reference_w=dict(Big=20,Little=8,FILL=0),workloads=workloads))
        manifest=dict(schema_version=1,base_commit=BASE_COMMIT,new_floorplan_count=count,
                      preserved_stage_a_count=23,preserved_pilot_count=5,total_floorplans=28+count,
                      expected_simulations=count*samples,parameters=params,floorplans=floorplans,
                      workload_manifest=wp.name,workload_manifest_sha256=sha256(wp),
                      temperature_labels='NOT GENERATED; real HotSpot execution required')
        save_json(temp/'floorplan_manifest.json',manifest)
        validate_dataset(temp/'floorplan_manifest.json')
        # Never replace a previously created destination, including concurrent generation.
        if output.exists():raise FileExistsError(output)
        temp.rename(output)
    finally:
        if temp.exists():shutil.rmtree(temp)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'dataset_v1')
    p.add_argument('--count',type=int,default=77)
    p.add_argument('--samples',type=int,default=20)
    p.add_argument('--seed',type=int,default=20261108)
    p.add_argument('--big-range',type=float,nargs=2,default=(10.,20.))
    p.add_argument('--little-range',type=float,nargs=2,default=(5.,10.))
    p.add_argument('--min-distance-um',type=int,default=150)
    p.add_argument('--validate-only',type=Path,metavar='MANIFEST')
    args=p.parse_args()
    if args.validate_only:
        manifest,jobs=validate_dataset(args.validate_only)
        print(f'VALID: {manifest["new_floorplan_count"]} new floorplans, {len(jobs)} workloads; no simulations')
    else:
        m=prepare(args.output,args.count,args.samples,args.seed,tuple(args.big_range),tuple(args.little_range),args.min_distance_um)
        print(f'Prepared {m["new_floorplan_count"]} new floorplans and {m["expected_simulations"]} power traces in {args.output}')

if __name__=='__main__':main()
