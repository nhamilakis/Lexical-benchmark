#!/usr/bin/env python


from lexical_benchmark.train import run_cmd

if __name__ == "__main__":
    cmd = run_cmd.Train.parse()
    cmd.start()
