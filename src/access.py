from lexical_benchmark import metadata

childes_meta: metadata.CHILDESMetaDir = metadata.get_config("childes", "EN")
childes_meta.builder.build_child_speech_quantities()
