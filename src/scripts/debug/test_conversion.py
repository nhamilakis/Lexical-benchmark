from lexical_benchmark import settings
from lexical_benchmark.dataloaders import generation_loaders

iterator = generation_loaders.GenerationCheckpointLoader.iter_items(datasets=("stela",))
root_dir = settings.PATH.generate_root / "text"

for item in iterator:
    if item.is_finished():
        checkpoint = item.load_final()
        for (estimation_type, month), obj in checkpoint.iter_items():
            target_item = generation_loaders.GenerationItemsLoader(
                estimation_type=estimation_type,
                model_type=item.model_type,
                lang=item.lang,
                month=month,
                model_chunk=f"{item.split}_{item.chunk}",
                temperature=item.temperature,
                dt_cfg=item.dt_cfg,
            )
            print(target_item)
            target_item.text_file.write_text("\n".join(obj["text"]))
