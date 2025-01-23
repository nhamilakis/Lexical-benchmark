from pathlib import Path

from lexical_benchmark import settings
from lexical_benchmark.datasets.gen_child.preparation import CHILDESMonth, ModelMonth,GenerationMerger


print('Testing GenerationMerger Class')

model_out_dir = settings.PATH.DATA_DIR /"gen"/ "merged"
model_processor = GenerationMerger(hour_per_year=1000,
                            gen_dir=model_out_dir)
model_processor.process()

print('Finished testing GenerationMerger Class')