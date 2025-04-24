#!/usr/bin/env python
import IPython

from lexical_benchmark.train import run_train

if __name__ == "__main__":
    cmd = run_train.Train.parse()
    if cmd.interactive:
        cmd_args = cmd.subcommand.prep_args()
        print("Using 'cmd: Command' & 'cmd_args: TrainArgsObj'")
        IPython.embed()
    else:
        cmd.start()
