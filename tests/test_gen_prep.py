from pathlib import Path

from lexical_benchmark import settings
from lexical_benchmark.datasets.gen_child.preparation import CHILDESMonth, ModelMonth,GenerationMerger


'''
print('Testing CHILDESMonth Class')
root_dir: Path = settings.PATH.dataset_root  / "CHILDES"
out_dir: Path = settings.PATH.DATA_DIR /"gen"/ "test"/ "CHILDES.csv"

childes_processor = CHILDESMonth(root_dir=root_dir,
                             out_dir=out_dir)
childes_processor.process()

print('Finished testing CHILDESMonth Class')




print('Testing ModelMonth Class')

model_out_dir = settings.PATH.DATA_DIR /"gen"/ "test"/ "CHILDES_model.csv"
model_processor = ModelMonth(child_df=out_dir,
                             out_dir=model_out_dir)
model_processor.process()

print('Finished testing ModelMonth Class')

'''

print('Testing GenerationMerger Class')

model_out_dir = settings.PATH.DATA_DIR /"gen"/ "merged"
model_processor = GenerationMerger(hour_per_year=1000,
                            gen_dir=model_out_dir)
model_processor.process()

print('Finished testing GenerationMerger Class')