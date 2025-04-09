#!/usr/bin/env python


from lexical_benchmark.train import run_generate

if __name__ == "__main__":
    cmd = run_generate.Generate.parse()
    cmd.start()
