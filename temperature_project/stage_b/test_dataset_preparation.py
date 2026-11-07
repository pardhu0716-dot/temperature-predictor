"""Dataset validation and simulation automation tests. HotSpot is always mocked."""
from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import uuid

from generate_floorplans import bounds, read_flp, validate
from generate_power_traces import (analyze_floorplan, generate_custom_trace, generate_random_traces,
    generate_reference_trace, GeometryValidationError, PowerAssignmentError)
from prepare_stage_b_dataset import HERE, PROJECT, prepare, validate_dataset, read_trace
import run_stage_b_simulations as runner

MANIFEST=HERE/'dataset_v1/floorplan_manifest.json'


def file_hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,cls.jobs=validate_dataset(MANIFEST)

    def test_full_dataset_geometry_and_power(self):
        self.assertEqual(self.manifest['new_floorplan_count'],77)
        self.assertEqual(self.manifest['total_floorplans'],105)
        self.assertEqual(len(self.jobs),1540)
        self.assertEqual(len({j['floorplan_id'] for j in self.jobs}),77)
        self.assertEqual(sum(j['sample_id']=='sample_001' for j in self.jobs),77)
        # Independent exact cell oracle, not the production summed-area check.
        for job in self.jobs[::20]:
            blocks=read_flp(job['flp']);w,h=bounds(blocks)
            xs=sorted({0,w,*[v for b in blocks for v in (b.x,b.x+b.w)]})
            ys=sorted({0,h,*[v for b in blocks for v in (b.y,b.y+b.h)]})
            self.assertEqual(sorted((b.w,b.h) for b in blocks if b.name.startswith('Core_')),
                             [(2000,2000)]*6+[(4000,4000)]*2)
            for x,x1 in zip(xs,xs[1:]):
                for y,y1 in zip(ys,ys[1:]):
                    cx=(x+x1)/2;cy=(y+y1)/2
                    self.assertEqual(sum(b.x<=cx<b.x+b.w and b.y<=cy<b.y+b.h for b in blocks),1)
        # All traces are validated by validate_dataset, including uniqueness and ranges.
        self.assertGreater(len({tuple(j['profile'].big_cores) for j in self.jobs}),10)
        for job in self.jobs:
            if job['sample_id']=='sample_001':
                self.assertEqual(sum(job['powers'].values()),88)
            for fill in job['profile'].fill_blocks:self.assertEqual(job['powers'][fill],0)

    def test_reproducible_generation_and_input_preservation(self):
        before_pilot=file_hashes(HERE/'pilot');before_source=file_hashes(PROJECT/'dataset')
        outputs=[HERE/f'.test-dataset-{uuid.uuid4().hex}' for _ in range(2)]
        for out in outputs:self.addCleanup(shutil.rmtree,out,True)
        for out in outputs:prepare(out,count=4,samples=5,seed=789)
        self.assertEqual(file_hashes(outputs[0]),file_hashes(outputs[1]))
        with self.assertRaises(FileExistsError):prepare(outputs[0],count=4,samples=5,seed=789)
        self.assertEqual(before_pilot,file_hashes(HERE/'pilot'))
        self.assertEqual(before_source,file_hashes(PROJECT/'dataset'))
        for out in [HERE/'pilot',PROJECT/'dataset/forbidden',HERE.parent/'forbidden']:
            with self.assertRaises((ValueError,FileExistsError)):prepare(out,count=1,samples=1)

    def test_trace_failures_and_nonfinite(self):
        job=self.jobs[0];profile=job['profile']
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.ptrace';headers=profile.header_order
            reference=generate_reference_trace(profile)
            for bad in [list(reversed(headers)),headers[:-1]]:
                path.write_text(' '.join(bad)+'\n'+' '.join(map(str,reference.powers))+'\n')
                with self.assertRaises(ValueError):read_trace(path,profile,(10,20),(5,10),True)
            for value in [1,float('nan'),float('inf')]:
                values=reference.powers[:];values[headers.index(profile.fill_blocks[0])]=value
                path.write_text(' '.join(headers)+'\n'+' '.join(map(str,values))+'\n')
                with self.assertRaises(ValueError):read_trace(path,profile,(10,20),(5,10),True)
            path=Path(tmp)/'bad.flp';path.write_text(job['flp'].read_text().replace('0.004000','nan',1))
            with self.assertRaises(GeometryValidationError):analyze_floorplan(path)
        powers={name:10 for name in profile.big_cores+profile.little_cores}
        for value in [float('nan'),float('inf'),-1]:
            with self.assertRaises(PowerAssignmentError):generate_custom_trace(profile,{**powers,profile.big_cores[0]:value})
        with self.assertRaises(PowerAssignmentError):generate_custom_trace(profile,powers,fill_power=1e-8)
        for ranges in [((float('nan'),20),(5,10)),((10,float('inf')),(5,10)),((10.001,10.002),(5,10))]:
            with self.assertRaises(ValueError):generate_random_traces(profile,2,big_range=ranges[0],little_range=ranges[1])
        with self.assertRaises(ValueError):generate_random_traces(profile,3,big_range=(20,20),little_range=(8,8))
        traces=generate_random_traces(profile,5,seed=123,big_range=(10.001,10.019),little_range=(5.001,5.029),include_reference=False)
        for trace in traces:
            for name in profile.big_cores:self.assertEqual(trace.power_dict()[name],10.01)
            for name in profile.little_cores:self.assertIn(trace.power_dict()[name],(5.01,5.02))

    def test_manifest_tamper_rejected(self):
        out=HERE/f'.test-dataset-{uuid.uuid4().hex}';self.addCleanup(shutil.rmtree,out,True)
        prepare(out,count=1,samples=2)
        path=next((out/'workloads').rglob('sample_002.ptrace'))
        path.write_text(path.read_text().replace('0.00','1.00'))
        with self.assertRaisesRegex(ValueError,'Changed ptrace'):validate_dataset(out/'floorplan_manifest.json')


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.job=validate_dataset(MANIFEST)[1][0]

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.output=self.root/'results'
        self.binary=self.root/'hotspot';self.binary.write_text('MOCK FILE ONLY — never executable simulator');self.binary.chmod(0o700)
        self.config=self.root/'hotspot.config';shutil.copyfile(PROJECT/'hotspot.config',self.config)
        self.names=self.job['profile'].big_cores+self.job['profile'].little_cores

    def fake_solver(self,command,**kwargs):
        # Demonstrate the solver sees isolated, resolvable layer files.
        cwd=Path(kwargs['cwd'])
        self.assertNotEqual(cwd,self.job['lcf'].parent)
        lcf=cwd/self.job['lcf'].name
        lines=[s for s in lcf.read_text().splitlines() if s and not s.startswith('#')]
        for i in (6,13):self.assertTrue((cwd/lines[i]).is_file())
        path=Path(command[command.index('-steady_file')+1])
        path.write_text('\n'.join(f'layer_0_{name} {300+i}' for i,name in enumerate(reversed(self.names)))+'\nlayer_0_FILL_0_L0 299\n')
        return subprocess.CompletedProcess(command,0)

    def test_command_and_selection_dry_run(self):
        command=runner.build_command(self.binary,self.config,self.job,self.root/'output.temp')
        self.assertEqual(command,[str(self.binary),'-c',str(self.config),'-f',self.job['flp'].name,'-p',str(self.job['ptrace']),
                                 '-grid_layer_file',self.job['lcf'].name,'-model_type','grid','-steady_file',str(self.root/'output.temp')])
        with patch.object(runner.subprocess,'run',side_effect=AssertionError('Dry-run must not invoke solver')):
            rc=runner.main(['--manifest',str(MANIFEST),'--output-dir',str(self.output),'--floorplans','b_fp_006',
                            '--samples','sample_001','--dry-run','--hotspot-bin','/does/not/exist'])
        self.assertEqual(rc,0);self.assertFalse(self.output.exists())
        self.assertEqual(runner.main(['--floorplans','missing','--dry-run']),1)
        with self.assertRaises(ValueError):runner.protect_output(HERE/'dataset_v1',MANIFEST)
        with self.assertRaises(ValueError):runner.protect_output(PROJECT/'dataset',MANIFEST)

    def test_kelvin_parser(self):
        path=self.root/'steady.temp'
        for prefix in ['', 'layer_0_']:
            path.write_text('\n'.join(f'{prefix}{name} 300' for name in reversed(self.names)))
            temperatures=runner.parse_temperatures(path,self.names)
            rows=list(runner.csv.DictReader(runner.io.StringIO(runner.label_text(self.job,temperatures))))
            self.assertEqual(len(rows),8)
            for row in rows:
                self.assertAlmostEqual(float(row['temperature_c']),26.85)
                self.assertEqual(float(row['power_w']),self.job['powers'][row['core_name']])
                self.assertEqual(row['core_type'],'Big' if row['core_name'] in self.job['profile'].big_cores else 'Little')
        valid='\n'.join(f'layer_0_{name} 300' for name in self.names)
        for text in ['\n'.join(valid.splitlines()[:-1]),valid+'\n'+valid.splitlines()[0],
                     valid.replace('300','nan',1),valid.replace('300','inf',1),valid.replace('300','-1',1)]:
            path.write_text(text)
            with self.assertRaises(ValueError):runner.parse_temperatures(path,self.names)

    def test_resume_and_overwrite_protection(self):
        with patch.object(runner.subprocess,'run',side_effect=self.fake_solver) as call:
            self.assertEqual(runner.run_job(self.job,self.binary,self.config,self.output),'success')
            self.assertEqual(call.call_count,1)
            self.assertEqual(runner.run_job(self.job,self.binary,self.config,self.output),'skipped')
            self.assertEqual(call.call_count,1)
        result=self.output/self.job['floorplan_id']/self.job['sample_id']/'result'
        labels=(result/'labels.csv').read_bytes()
        self.config.write_text(self.config.read_text()+'\n# changed config\n')
        with self.assertRaises(FileExistsError):runner.run_job(self.job,self.binary,self.config,self.output)
        self.assertEqual(labels,(result/'labels.csv').read_bytes())
        fp=runner.input_fingerprint(self.job,self.config,self.binary)[0]
        self.assertFalse(runner.result_valid(result,self.job,fp))

    def test_read_only_result_validation(self):
        args=['--manifest',str(MANIFEST),'--output-dir',str(self.output),'--limit','1',
              '--hotspot-bin',str(self.binary),'--config',str(self.config),'--validate-results']
        with patch.object(runner.subprocess,'run',side_effect=AssertionError('Validation must not simulate')):
            self.assertEqual(runner.main(args),1)
        self.assertFalse(self.output.exists())
        with patch.object(runner.subprocess,'run',side_effect=self.fake_solver):
            runner.run_job(self.job,self.binary,self.config,self.output)
        before=file_hashes(self.output)
        with patch.object(runner.subprocess,'run',side_effect=AssertionError('Validation must not simulate')):
            self.assertEqual(runner.main(args),0)
        self.assertEqual(before,file_hashes(self.output))

    def test_tampered_labels_and_binary_rejected(self):
        with patch.object(runner.subprocess,'run',side_effect=self.fake_solver):runner.run_job(self.job,self.binary,self.config,self.output)
        result=self.output/self.job['floorplan_id']/self.job['sample_id']/'result'
        (result/'labels.csv').write_text('corrupt')
        with self.assertRaises(FileExistsError):runner.run_job(self.job,self.binary,self.config,self.output)
        self.binary.write_text('new mock binary')
        with self.assertRaises(FileExistsError):runner.run_job(self.job,self.binary,self.config,self.output)

    def test_failure_and_retry_no_labels(self):
        with patch.object(runner.subprocess,'run',return_value=subprocess.CompletedProcess([],2)):
            with self.assertRaises(RuntimeError):runner.run_job(self.job,self.binary,self.config,self.output)
        folder=self.output/self.job['floorplan_id']/self.job['sample_id']
        self.assertFalse((folder/'result').exists());self.assertFalse((folder/'.running.lock').exists())
        self.assertFalse(list(folder.rglob('labels.csv')))
        attempt=list((folder/'attempts').glob('*'))[0]
        self.assertEqual(json.loads((attempt/'run.json').read_text())['status'],'failed')
        with patch.object(runner.subprocess,'run',side_effect=self.fake_solver):
            self.assertEqual(runner.run_job(self.job,self.binary,self.config,self.output),'success')
        self.assertEqual(len(list((folder/'attempts').glob('*'))),2)
        self.assertEqual(json.loads((attempt/'run.json').read_text())['status'],'failed')

    def test_timeout_missing_output_and_lock(self):
        for effect in [subprocess.TimeoutExpired('mock',1),subprocess.CompletedProcess([],0)]:
            folder=self.root/f'case_{uuid.uuid4().hex}'
            kwargs={'side_effect':effect} if isinstance(effect,Exception) else {'return_value':effect}
            with patch.object(runner.subprocess,'run',**kwargs):
                with self.assertRaises((subprocess.TimeoutExpired,FileNotFoundError)):runner.run_job(self.job,self.binary,self.config,folder)
            self.assertFalse(list(folder.rglob('labels.csv')))
        folder=self.output/self.job['floorplan_id']/self.job['sample_id'];folder.mkdir(parents=True)
        (folder/'.running.lock').write_text('test lock')
        with self.assertRaises(RuntimeError):runner.run_job(self.job,self.binary,self.config,self.output)

if __name__=='__main__':unittest.main()
