#!/usr/bin/env python


from lexical_benchmark.train import run_train

if __name__ == "__main__":
    cmd = run_train.Train.parse()
    cmd.start()
