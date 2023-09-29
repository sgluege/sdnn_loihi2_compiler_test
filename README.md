# sdnn_loihi2_compiler_test
Script to test compilation of a given SDNN model for loihi2


## Requirements / Test environment
- python 3.8.10
- lava_loihi-0.5.0
- lava_dl-0.4.0
- lava_nc-0.8.0

## Description
The script `compiler_test_trained_cnn_encoder.py` loads a model (`data/model.net`) which was exported using the 'export_hdf5' function of the SDNN slayer blocks and compiles it for the Loihi2 neuromorphic hardware.

```python
def export_hdf5(self, filename):
    # network export to hdf5 format
    h = h5py.File(filename, 'w')
    layer = h.create_group('layer')
    for i, b in enumerate(self.blocks):
        b.export_hdf5(layer.create_group(f'{i}'))
```

Output example:
```
|   Type   |  W  |  H  |  C  | ker | str | pad | dil | grp |delay|
|Conv      |  172|  130|    4| 3, 3| 2, 2| 1, 1| 1, 1|    1|False|
|Conv      |   86|   65|    8| 3, 3| 2, 2| 2, 2| 2, 2|    1|False|
|Conv      |   43|   33|    8| 3, 3| 2, 2| 2, 2| 2, 2|    1|False|
|Dense     |    1|    1| 1000|     |     |     |     |     |False|
|Dense     |    1|    1|  250|     |     |     |     |     |False|
|Dense     |    1|    1|    6|     |     |     |     |     |False|
There are 6 layers in the network:
Conv  : Process_3 , shape : (172, 130, 4)
Conv  : Process_6 , shape : (86, 65, 8)
Conv  : Process_9 , shape : (43, 33, 8)
Dense : Process_12, shape : (1000,)
Dense : Process_15, shape : (250,)
Dense : Process_18, shape : (6,)
compiling network
```

## Problem
The compilation of the model fails with the following error message:
```
---------------------------------------------------------------------------
Exception                                 Traceback (most recent call last)
/home/sdnn_loihi2_compiler_test/compiler_test_trained_cnn_encoder.py in line 29
     26 compile_only = True
     27 if compile_only:
     28     # Compile but do not run the network
---> 29     exec = net.compile(run_config, None)
     30     print('WE ARE DONE')
     31 else:
     32     # This will try to execute the dummy network

File /usr/local/lib/python3.8/dist-packages/lava/magma/core/process/process.py:398, in AbstractProcess.compile(self, run_cfg, compile_config)
    396 from lava.magma.compiler.compiler import Compiler
    397 compiler = Compiler(compile_config, self._log_config.level)
--> 398 return compiler.compile(self, run_cfg)

File /usr/local/lib/python3.8/dist-packages/lava/magma/compiler/compiler.py:134, in Compiler.compile(self, process, run_cfg)
    132 process_list = list(itertools.chain.from_iterable(proc_groups))
    133 channel_map = ChannelMap.from_proc_groups(proc_groups)
--> 134 proc_builders, channel_map = self._compile_proc_groups(
    135     proc_groups, channel_map
    136 )
    137 _, c_builders, nc_builders = split_proc_builders_by_type(
    138     proc_builders
    139 )
...
--> 211     raise Exception("Partitioning could not converge")
    212 d = DiscreteHyperParam(num_neurons_per_core=num_neurons_per_core)
    213 self._hyper_params.append(d)

Exception: Partitioning could not converge
```
