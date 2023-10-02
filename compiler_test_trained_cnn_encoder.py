import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from lava.utils.system import Loihi2
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg, Loihi2HwCfg
from lava.magma.compiler.subcompilers.nc.ncproc_compiler import CompilerOptions
from lava.proc import io
from lava.lib.dl import netx
import h5py

project_path = './'
model_path = project_path + 'data/'
# model_name = 'model.net'
# model_name = 'model_1hot.net'
model_name = 'model_heatmaps.net'

if __name__ == "__main__":
    CompilerOptions.verbose = True
    CompilerOptions.log_memory_info = True
    CompilerOptions.show_resource_count = True

    net = netx.hdf5.Network(net_config=model_path + model_name, skip_layers=1)
    print(net)
    print(f'There are {len(net)} layers in the network:')
    for l in net.layers:
        print(f'{l.__class__.__name__:5s} : {l.name:10s}, shape : {l.shape}')
    
    run_config = Loihi2HwCfg()
    print('compiling network')

    compile_only = True

    if compile_only:
        # Compile but do not run the network
        exec = net.compile(run_config, None)
        print('WE ARE DONE')
    else:
        # This will try to execute the dummy network
        net._log_config.level = logging.INFO
        net.run(condition=RunSteps(num_steps=10), run_cfg=run_config)
        net.stop()
