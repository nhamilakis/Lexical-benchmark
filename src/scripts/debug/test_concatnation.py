from lexical_benchmark import settings

from lexical_benchmark.dataloaders import generation_loaders


def get_path():
    """Get ckpt path from the object."""

# Simple debug version of your loop
iterator = generation_loaders.GenerationCheckpointLoader.iter_items(
    datasets=("stela",)
)

total_items = 0
finished_items = 0
written_files = 0

fail_lst = []
success_lst = []
for item in iterator:
    total_items += 1
    if item.is_finished():
        finished_items += 1
        checkpoint = item.load_final()
        
        checkpoint_items = list(checkpoint.iter_items())

        # load pickle object


        if len(checkpoint_items) == 0:
            #print(f"Failed to load file from {item}: {checkpoint}")
            fail_lst.append(item)    
        else:
            success_lst.append(item)    

        # loop different actual months
        for checkpoint_item in checkpoint_items:
            estimation_type, month, obj = checkpoint_item
            
            target_item = generation_loaders.GenerationItemsLoader.load(
                dataset_name=item.dt_cfg.dataset_name,
                estimation_type=estimation_type,
                model_type=item.model_type,
                lang=item.lang,
                month=month,
                model_chunk=f"{item.split}_{item.chunk}",
                temperature=item.temperature,
            )
            # Ensure directory exists
            target_item.text_file.parent.mkdir(parents=True, exist_ok=True)
            # Write text safely
            target_item.text_file.write_text("\n".join(obj["text"]))
            written_files += 1
            
  

print(f"\nSummary:")
print(f"Total items: {total_items}")
print(f"Finished items: {finished_items}")
print(f"Written files: {written_files}")





