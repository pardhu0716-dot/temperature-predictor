"""Run from temperature_project: python -m stage_b.ml --help."""
import argparse
import csv
import json
from pathlib import Path
import sys
from .data import load_csv,read_json,schema_template
from .preprocessing import prepare,safe_output
from .experiments import configurations
from .training import train,evaluate_test
from .external import check_compatibility


def main(argv=None):
    p=argparse.ArgumentParser(description='Stage B geometry-aware multi-output ML infrastructure; never runs HotSpot')
    sub=p.add_subparsers(dest='command',required=True)
    inspect=sub.add_parser('inspect-csv',help='Print actual local header/counts without guessing mappings')
    inspect.add_argument('--csv',type=Path,required=True)
    template=sub.add_parser('schema-template',help='Write EXAMPLE column mapping; edit to the actual inspected header')
    template.add_argument('--out',type=Path,required=True)
    for name in ('validate','prepare'):
        cmd=sub.add_parser(name)
        cmd.add_argument('--csv',type=Path,required=True);cmd.add_argument('--schema',type=Path,required=True)
        cmd.add_argument('--geometry-manifest',type=Path,default=Path('stage_b/dataset_v1/floorplan_manifest.json'))
        if name=='prepare':
            cmd.add_argument('--out',type=Path,required=True);cmd.add_argument('--seed',type=int,default=42)
            cmd.add_argument('--split-counts',type=int,nargs=3,default=(53,12,12))
    cmd=sub.add_parser('train',help='Train/compare on train+validation only; never opens test.npz')
    cmd.add_argument('--prepared',type=Path,required=True);cmd.add_argument('--out',type=Path,required=True)
    cmd.add_argument('--group',choices=['core','optimizers','regularization'],default='core')
    cmd.add_argument('--models',nargs='+',help='Exact experiment names in group')
    cmd.add_argument('--experiments',type=Path,help='JSON list of {name,kind,parameters}; overrides group')
    cmd.add_argument('--seed',type=int,default=42);cmd.add_argument('--latency-repeats',type=int,default=50)
    for name,typ in [('epochs',int),('learning-rate',float),('batch-size',int),('tolerance',float)]:cmd.add_argument('--'+name,type=typ)
    cmd.add_argument('--alpha',type=float,help='Ridge/Lasso/Elastic Net penalty override')
    cmd=sub.add_parser('evaluate-test',help='Explicit final holdout evaluation; selected model only; no overwrite')
    cmd.add_argument('--prepared',type=Path,required=True);cmd.add_argument('--run',type=Path,required=True)
    cmd=sub.add_parser('external-check',help='Validate future external simulation/feature contracts only; no evaluation')
    cmd.add_argument('--training-contract',type=Path,required=True);cmd.add_argument('--external-contract',type=Path,required=True)
    args=p.parse_args(argv)
    try:
        if args.command=='inspect-csv':
            with args.csv.open(newline='',encoding='utf-8-sig') as stream:
                reader=csv.reader(stream);header=next(reader,None)
                if not header:raise ValueError('CSV is empty or has no header')
                rows=sum(1 for _ in reader)
            print(json.dumps(dict(rows=rows,column_count=len(header),columns=header),indent=2));return 0
        if args.command=='schema-template':
            path=safe_output(args.out);path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('x') as stream:json.dump(schema_template(),stream,indent=2)
            print('EXAMPLE mapping created. Inspect CSV and edit all column/unit declarations before validation.');return 0
        if args.command in ('validate','prepare'):
            data=load_csv(args.csv,read_json(args.schema),args.geometry_manifest)
            if args.command=='validate':print(json.dumps(dict(rows=len(data.floorplan_ids),floorplans=len(set(data.floorplan_ids)),targets=8,status='validated; label provenance declared by schema, not independently reconstructed')))
            else:print(json.dumps(prepare(data,args.out,args.seed,tuple(args.split_counts),args.geometry_manifest),indent=2))
        elif args.command=='train':
            experiments=read_json(args.experiments) if args.experiments else configurations(args.group)
            if args.models:
                missing=set(args.models)-{e['name'] for e in experiments}
                if missing:raise ValueError(f'Unknown experiment selection: {sorted(missing)}')
                experiments=[e for e in experiments if e['name'] in args.models]
            for flag in ('epochs','learning_rate','batch_size','tolerance','alpha'):
                value=getattr(args,flag)
                if value is None:continue
                kinds=('ridge','lasso','elastic_net') if flag=='alpha' else ('linear_gradient','mlp')
                matching=[e for e in experiments if e['kind'] in kinds]
                if not matching:raise ValueError(f'{flag} does not apply to selected estimators')
                for e in matching:e.setdefault('parameters',{})[flag]=value
            results=train(args.prepared,args.out,experiments,args.seed,args.latency_repeats)
            print(json.dumps([dict(name=r['experiment']['name'],validation=r['validation'],train_seconds=r['train_seconds'],serialized_model_bytes=r['serialized_model_bytes']) for r in results],indent=2))
        elif args.command=='evaluate-test':print(json.dumps(evaluate_test(args.prepared,args.run),indent=2))
        elif args.command=='external-check':print(json.dumps(check_compatibility(read_json(args.training_contract),read_json(args.external_contract)),indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError,ImportError,RuntimeError,FloatingPointError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr);return 1

if __name__=='__main__':sys.exit(main())
