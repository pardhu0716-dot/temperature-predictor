# Sample Dataset — Lightweight Temperature Predictor

Sample floorplans and power traces in HotSpot format, for training models or as templates for generating more data.

## Layout

```
2D_floorplans/
  Style_1/ … Style_4/        floorplan generation styles (Style_4 has 10 samples, others 3–5)
    fp_001/
      fp_001.lcf             layer configuration file (HotSpot -grid_layer_file)
      fp_001_layer_0.flp     silicon (core) layer
      fp_001_tim.flp         thermal interface material layer
      fp_001_steady.ptrace     steady power trace (W per block)
      fp_001_transient.ptrace  transient power trace (200 rows, one per sampling interval)
      fp_001_layer_0.png     floorplan plot
3D_floorplans/
  2_layer/fp_001 … fp_005/   two core tiers: layer_0/1.flp, tim.flp, .lcf, plots 
  4_layer/fp_001 … fp_009/   four core tiers: layer_0..3.flp, tim.flp (+ tim_tsv.flp in fp_004–009), .lcf, plots
                             fp_004–009 include a _steady.ptrace (big core 20 W, little core 8 W); fp_001–003 include 3d_stack 
```
The multi-core floorplan varies in number of core types, size of cores, and the size of the overall die/chip. The cores are active Si regions which dissipates power in the form of heat.
The white space in a chip is a passive area which does not consumes any power (0W). It is kept in a chip with some allowed % to keep the System-on-Chip (SoC) cool.


## Power Trace Inputs

The floorplan directories consist of the ptrace in the format which the HotSpot tool supports. However, the power inputs for big/little cores can be different for different workloads. 
For the experimental exploration, you can vary the power inputs by hardcoding it. (in future, you can run a workload on different cores to get the real power data using standard tools like **turbostat** or **Intel RAPL**)
For P/Big cores, (you can choose power numbers within range (10-20W))
For E/Little cores, (you can choose power numbers within range (5-10W))
Note: These numbers are not realistic, just a practical assumption for practice, knowing the convention that Big cores consumes higher power than little cores.

## File formats

- **.flp** — one block per line: `name  width  height  left-x  bottom-y` (metres). Lines starting with `#` are comments. Every layer covers the full die; unused area is made of zero-power `FILL_*` blocks.
- **.lcf** — 7 lines per layer: layer number, lateral heat flow (Y/N), power dissipation (Y/N), specific heat (J/m³K), resistivity (mK/W), thickness (m), floorplan file. Layer 0 is the bottom; the last layer is next to the heat spreader. Floorplan paths are relative to the folder.
- **.ptrace** — first line: block names; each following line: power (W) per block. Columns follow the block order of the power-dissipating layers in the .lcf.
  In transient mode (`-o out.ttrace`) HotSpot treats each row as one `sampling_intvl` step, so power may change from row to row; the steady-state result uses the average power over all rows.

## Running HotSpot

From inside a floorplan folder, with a HotSpot config file `hotspot.config` (64×64 grid):

```
hotspot -c hotspot.config -f fp_001_layer_0.flp -p fp_001_steady.ptrace \
        -grid_layer_file fp_001.lcf -model_type grid -steady_file fp_001.steady
```

The sink and spreader in the config must be larger than the die (the largest die here is 16.5 mm).
