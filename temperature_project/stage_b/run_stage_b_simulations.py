#!/usr/bin/env python3
"""Automate real HotSpot steady-state grid runs on WSL/Linux; no thermal model here."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from prepare_stage_b_dataset import (HERE, PROJECT, save_json, sha256, validate_dataset)

LOG=logging.getLogger('stage_b')
LABEL_FIELDS=['floorplan_id','sample_id','core_name','core_type','power_w','temperature_k','temperature_c']
RUNNER_VERSION=1


def parse_temperatures(path, core_names):
    """Read HotSpot block steady output; accept raw names or grid layer_0_ prefixes.

    Missing, repeated, nonfinite, and below-zero Kelvin core values fail closed.
    Passive blocks and other layers are ignored, never used as core labels.
    """
    expected=set(core_names);result={}
    for number,line in enumerate(Path(path).read_text().splitlines(),1):
        fields=line.split()
        if not fields or fields[0].startswith('#'):continue
        name=fields[0]
        if name.startswith('layer_0_'):name=name[len('layer_0_'):]
        if name not in expected:continue
        if len(fields)!=2:raise ValueError(f'Malformed core temperature at line {number}')
        value=float(fields[1])
        if not math.isfinite(value) or value < 0:raise ValueError(f'Invalid Kelvin temperature: {name}')
        if name in result:raise ValueError(f'Duplicate temperature: {name}')
        result[name]=value
    if set(result)!=expected:raise ValueError(f'Missing core temperatures: {sorted(expected-set(result))}')
    return result


def label_text(job, temperatures):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=LABEL_FIELDS,lineterminator='\n')
    writer.writeheader()
    for name in job['profile'].header_order:
        if name not in temperatures:continue
        k=temperatures[name]
        writer.writerow(dict(floorplan_id=job['floorplan_id'],sample_id=job['sample_id'],core_name=name,
                            core_type='Big' if name in job['profile'].big_cores else 'Little',
                            power_w=job['powers'][name],temperature_k=format(k,'.12g'),temperature_c=format(k-273.15,'.12g')))
    return stream.getvalue()


def build_command(hotspot, config, job, steady):
    return [str(hotspot),'-c',str(config),'-f',job['flp'].name,'-p',str(job['ptrace']),
            '-grid_layer_file',job['lcf'].name,'-model_type','grid','-steady_file',str(steady)]


def input_fingerprint(job, config, hotspot):
    """Include binary, runner, config, layer files, geometry, and power contents."""
    files={p.name:sha256(p) for p in [job['lcf'],*job['layers'],job['ptrace']]}
    identity=dict(runner_version=RUNNER_VERSION,runner_sha256=sha256(Path(__file__)),
                  floorplan_id=job['floorplan_id'],sample_id=job['sample_id'],
                  files=files,config_sha256=sha256(config),hotspot_sha256=sha256(hotspot),
                  model_type='grid',temperature_units='Kelvin; Celsius=Kelvin-273.15')
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),identity


def result_valid(result, job, fingerprint):
    """Only a successful result with intact raw output and exact labels may resume."""
    try:
        info=json.loads((result/'run.json').read_text())
        if info['status']!='success' or info['fingerprint']!=fingerprint:return False
        if sha256(result/'steady.temp')!=info['steady_sha256']:return False
        if sha256(result/'labels.csv')!=info['labels_sha256']:return False
        temperatures=parse_temperatures(result/'steady.temp',job['profile'].big_cores+job['profile'].little_cores)
        return (result/'labels.csv').read_text()==label_text(job,temperatures)
    except (OSError,ValueError,KeyError,TypeError):return False


def protect_output(output,manifest):
    output=Path(output).expanduser().resolve()
    # Repository-local outputs are allowed only within the separate results tree.
    if output.is_relative_to(PROJECT.parent) and not output.is_relative_to(HERE/'simulation_results'):
        raise ValueError('Repository-local simulation output must be inside stage_b/simulation_results')
    protected=[PROJECT/'dataset',HERE/'pilot',Path(manifest).resolve().parent]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Output overlaps protected inputs')
    return output


def run_job(job, hotspot, config, output, timeout=300):
    fingerprint,identity=input_fingerprint(job,config,hotspot)
    folder=output/job['floorplan_id']/job['sample_id'];result=folder/'result'
    if result.exists():
        if result_valid(result,job,fingerprint):
            LOG.info('SKIP %s/%s: verified existing result',job['floorplan_id'],job['sample_id']);return 'skipped'
        raise FileExistsError(f'Existing result is invalid or inputs changed: {result}. Use a new output directory; it was not overwritten.')
    folder.mkdir(parents=True,exist_ok=True)
    lock=folder/'.running.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as exc:raise RuntimeError(f'Job locked: {lock}; check for another process or a stale lock') from exc
    with os.fdopen(fd,'w') as stream:stream.write(f'pid={os.getpid()}\n')
    attempt=folder/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
    started=datetime.now(timezone.utc).isoformat();t0=time.monotonic()
    info=dict(status='running',fingerprint=fingerprint,inputs=identity,started_utc=started,
              floorplan_id=job['floorplan_id'],sample_id=job['sample_id'])
    try:
        # Isolate incidental HotSpot output; preserve LCF relative paths in this workdir.
        workspace=attempt/'workspace';workspace.mkdir()
        for source in [job['lcf'],*job['layers']]:
            relative=source.relative_to(job['lcf'].parent)
            target=workspace/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        steady=attempt/'steady.temp'
        command=build_command(hotspot,config,job,steady)
        info.update(command=command,cwd=str(workspace))
        save_json(attempt/'run.json',info)
        LOG.info('RUN %s/%s',job['floorplan_id'],job['sample_id'])
        with (attempt/'stdout.log').open('w') as stdout,(attempt/'stderr.log').open('w') as stderr:
            completed=subprocess.run(command,cwd=workspace,stdout=stdout,stderr=stderr,timeout=timeout,check=False)
        info['returncode']=completed.returncode
        if completed.returncode != 0:raise RuntimeError(f'HotSpot exit {completed.returncode}; see {attempt}/stderr.log')
        temperatures=parse_temperatures(steady,job['profile'].big_cores+job['profile'].little_cores)
        # Catch input/config/binary changes while the solver was running.
        if input_fingerprint(job,config,hotspot)[0]!=fingerprint:raise RuntimeError('Inputs changed during simulation')
        labels=label_text(job,temperatures)
        info.update(status='success',elapsed_seconds=time.monotonic()-t0,
                    steady_sha256=sha256(steady),labels_sha256=hashlib.sha256(labels.encode()).hexdigest())
        save_json(attempt/'run.json',info)
        publish=attempt/'publish';publish.mkdir()
        shutil.copyfile(steady,publish/'steady.temp');(publish/'labels.csv').write_text(labels)
        save_json(publish/'run.json',info)
        # A failed/interrupted attempt cannot masquerade as a resumable result.
        if result.exists():raise FileExistsError(f'Result appeared concurrently: {result}')
        publish.rename(result)
        return 'success'
    except Exception as exc:
        info.update(status='failed',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.monotonic()-t0)
        save_json(attempt/'run.json',info)
        raise
    finally:
        lock.unlink(missing_ok=True)


def config_check(config, jobs):
    if not config.is_file():raise FileNotFoundError(f'HotSpot config not found: {config}')
    options={}
    for line in config.read_text().splitlines():
        parts=line.split('#',1)[0].split()
        if len(parts)>=2:options[parts[0]]=parts[1]
    for key in ('-s_spreader','-s_sink'):
        side=float(options[key])
        if not math.isfinite(side) or side<=max(max(job['profile'].die_width,job['profile'].die_height) for job in jobs):
            raise ValueError(f'{key} must exceed both die dimensions')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=HERE/'dataset_v1/floorplan_manifest.json')
    p.add_argument('--hotspot-bin',type=Path,default=Path('~/HotSpot/hotspot'))
    p.add_argument('--config',type=Path,default=PROJECT/'hotspot.config')
    p.add_argument('--output-dir',type=Path,default=HERE/'simulation_results/dataset_v1')
    p.add_argument('--floorplans',nargs='+',help='Exact IDs, e.g. b_fp_006 b_fp_007')
    p.add_argument('--samples',nargs='+',help='Exact IDs, e.g. sample_001 sample_002')
    p.add_argument('--limit',type=int,help='Maximum selected jobs (deterministic manifest order)')
    p.add_argument('--timeout',type=float,default=300,help='Per-job timeout in seconds')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true',help='Validate and print commands; no execution or output writes')
    mode.add_argument('--validate-results',action='store_true',help='Read-only check of existing results; never executes HotSpot')
    args=p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    try:
        if not math.isfinite(args.timeout) or args.timeout<=0:raise ValueError('Timeout must be finite and positive')
        if args.limit is not None and args.limit<1:raise ValueError('Limit must be positive')
        manifest,jobs=validate_dataset(args.manifest)
        if args.floorplans:
            unknown=set(args.floorplans)-{j['floorplan_id'] for j in jobs}
            if unknown:raise ValueError(f'Unknown floorplan selection: {sorted(unknown)}')
            jobs=[j for j in jobs if j['floorplan_id'] in args.floorplans]
        if args.samples:
            unknown=set(args.samples)-{j['sample_id'] for j in jobs}
            if unknown:raise ValueError(f'Unknown sample selection: {sorted(unknown)}')
            jobs=[j for j in jobs if j['sample_id'] in args.samples]
        if args.limit:jobs=jobs[:args.limit]
        if not jobs:raise ValueError('No jobs selected')
        config=args.config.expanduser().resolve();hotspot=args.hotspot_bin.expanduser().resolve()
        config_check(config,jobs)
        output=protect_output(args.output_dir,args.manifest)
        if args.dry_run:
            for job in jobs:
                command=build_command(hotspot,config,job,output/job['floorplan_id']/job['sample_id']/'attempts/<attempt>/steady.temp')
                print(json.dumps(dict(floorplan=job['floorplan_id'],sample=job['sample_id'],
                                      command=command,cwd='isolated copy of '+str(job['lcf'].parent))))
            LOG.info('DRY RUN: %d jobs; no simulations or output writes',len(jobs));return 0
        if not hotspot.is_file() or not os.access(hotspot,os.X_OK):
            raise FileNotFoundError(f'HotSpot executable not found/executable: {hotspot}; use --hotspot-bin')
        if args.validate_results:
            valid=0
            for job in jobs:
                result=output/job['floorplan_id']/job['sample_id']/'result'
                fingerprint,_=input_fingerprint(job,config,hotspot)
                if result_valid(result,job,fingerprint):valid+=1
                else:LOG.error('MISSING/INVALID %s/%s: %s',job['floorplan_id'],job['sample_id'],result)
            LOG.info('RESULT VALIDATION: %d valid, %d missing/invalid; no simulations or output writes',valid,len(jobs)-valid)
            return 0 if valid==len(jobs) else 1
        stats=dict(success=0,skipped=0,failed=0)
        for job in jobs:
            try:stats[run_job(job,hotspot,config,output,args.timeout)]+=1
            except Exception as exc:
                stats['failed']+=1;LOG.error('%s/%s: %s',job['floorplan_id'],job['sample_id'],exc)
        output.mkdir(parents=True,exist_ok=True)
        save_json(output/f'batch_{uuid.uuid4().hex}.json',dict(manifest=str(args.manifest.resolve()),selection=dict(
            floorplans=args.floorplans,samples=args.samples,limit=args.limit),counts=stats,finished_utc=datetime.now(timezone.utc).isoformat()))
        LOG.info('Completed: %s',stats)
        return 1 if stats['failed'] else 0
    except (OSError,ValueError,KeyError,TypeError) as exc:
        LOG.error('%s: %s',type(exc).__name__,exc);return 1

if __name__=='__main__':sys.exit(main())
