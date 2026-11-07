"""Run: python3 -m unittest discover -s temperature_project/stage_b -v"""
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest
from generate_floorplans import (Block, SOURCE, catalog, descriptor, distance,
    fill_whitespace, generate, parser, read_flp, validate, write_flp)

HERE=Path(__file__).resolve().parent

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cores=[Block(f'Core_{i}_L0',4000 if i<2 else 2000,4000 if i<2 else 2000,
                          i*5000,0) for i in range(8)]

    def assert_cells(self, blocks,w,h):
        # Independent coverage oracle: every induced cell covered exactly once.
        xs=sorted({0,w,*[v for b in blocks for v in (b.x,b.x+b.w)]})
        ys=sorted({0,h,*[v for b in blocks for v in (b.y,b.y+b.h)]})
        for x,x1 in zip(xs,xs[1:]):
            for y,y1 in zip(ys,ys[1:]):
                cx=(x+x1)/2;cy=(y+y1)/2
                self.assertEqual(sum(b.x<=cx<b.x+b.w and b.y<=cy<b.y+b.h for b in blocks),1)

    def test_original_catalog(self):
        entries=catalog(SOURCE)
        self.assertEqual([sum(e['style']==f'Style_{i}' for e in entries) for i in range(1,5)],[3,5,5,10])
        for e in entries:self.assert_cells(read_flp(e['path']),e['width'],e['height'])

    def test_fill_and_roundtrip(self):
        blocks=fill_whitespace(self.cores,40000,6000)
        validate(blocks,40000,6000);self.assert_cells(blocks,40000,6000)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'test.flp';write_flp(p,blocks)
            self.assertEqual(read_flp(p),blocks)
            self.assertTrue(all(len(line.split())==5 for line in p.read_text().splitlines() if not line.startswith('#')))

    def test_invalid_geometries(self):
        for invalid in [self.cores+[self.cores[0]],self.cores[1:],
                        [Block('Core_0_L0',4000,4000,-1,0)]+self.cores[1:],
                        [Block('Core_0_L0',0,4000,0,0)]+self.cores[1:],
                        [Block('Core_0_L0',4000,4000,5000,0)]+self.cores[1:]]:
            with self.assertRaises(ValueError):validate(invalid,40000,6000,complete=False)
        with self.assertRaises(ValueError):validate(self.cores,40000,6000)
        with self.assertRaises(ValueError):validate(self.cores,1000,1000,complete=False)
        with self.assertRaises(ValueError):validate(self.cores,40000,6000,complete=False,gap=4000)

    def test_malformed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'bad.flp'
            for text in ['a 1 2 3','a nan 1 0 0','a 0.0000001 1 0 0']:
                p.write_text(text)
                with self.assertRaises(ValueError):read_flp(p)

    def test_diversity_invariance(self):
        a=descriptor(self.cores,40000,6000)
        mirrored=[Block(b.name,b.w,b.h,40000-b.x-b.w,b.y) for b in reversed(self.cores)]
        self.assertEqual(distance(a,descriptor(mirrored,40000,6000)),0)

    def test_pilot(self):
        entries=catalog(SOURCE);seen=[e['descriptor'] for e in entries]
        manifest=json.loads((HERE/'pilot/manifest.json').read_text())
        self.assertEqual(len(manifest['floorplans']),5)
        for m in manifest['floorplans']:
            folder=HERE/'pilot'/m['id'];name=m['id'];w,h=m['die_um']
            blocks=read_flp(folder/f'{name}_layer_0.flp');validate(blocks,w,h)
            self.assert_cells(blocks,w,h)
            validate(read_flp(folder/f'{name}_tim.flp'),w,h,cores=False)
            desc=descriptor([b for b in blocks if b.name.startswith('Core_')],w,h)
            self.assertGreaterEqual(min(distance(desc,d) for d in seen),150);seen.append(desc)
            lcf=[s for s in (folder/f'{name}.lcf').read_text().splitlines() if s and not s.startswith('#')]
            self.assertEqual(len(lcf),14)
            for i in [6,13]:self.assertTrue((folder/lcf[i]).is_file())
            trace=(folder/f'{name}_zero.ptrace').read_text().splitlines()
            self.assertEqual(trace[0].split(),[b.name for b in blocks])
            self.assertEqual(len(trace[1].split()),len(blocks))
            self.assertTrue(all(float(v)==0 for v in trace[1].split()))

    def test_reproducible_and_safe_output(self):
        # Temporary outputs are cleaned up; only five pilot floorplans are retained.
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            out=Path(temp)/'repeat'
            args=parser().parse_args(['--output',str(out)])
            generate(args)
            for source in (HERE/'pilot').glob('b_fp_*/*'):
                if source.suffix in ('.flp','.lcf','.ptrace'):
                    self.assertEqual(source.read_bytes(),(out/source.parent.name/source.name).read_bytes())
            with self.assertRaises(FileExistsError):generate(args)
        args=parser().parse_args(['--output',str(SOURCE/'do_not_write')])
        with self.assertRaises(ValueError):generate(args)

if __name__=='__main__':unittest.main()
