#!/usr/bin/env python


from clypi import ClypiConfig, configure

from lexical_benchmark.train import run_generate

# Test configurations

configure(ClypiConfig(nice_errors=()))

if __name__ == "__main__":
    cmd = run_generate.Generate.parse()  # Parsing CMD args
    if cmd.interactive:
        import IPython

        data_item, generator, token_nb_mapping = cmd.subcommand.prep_args()
        print("""Using:
        - 'cmd: Command'
        - 'data_item: GeneratedItem'
        - 'generator: ModelClass'
        - 'token_nb_mapping<dict>: Mapping of Required Tokens'
        """)
        IPython.embed()
    else:
        cmd.start()
