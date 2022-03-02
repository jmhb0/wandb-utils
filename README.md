Example usage 

```
import wandb_utils.wandb_utils as wa
url="https://wandb.ai/jmhb0/allen-10000-56/runs/26s3qdw9?workspace=user-jmhb0"
entity, project, run_id = wa.run_meta_from_url(url)
print(entity, project, run_id, )
# Returns: "jmhb0" "allen-10000-56" "26s3qdw9"

run=wa.get_run(entity, project, run_id)
files, total_size = wa.get_run_files(run, print_sz=0)
print(f"Run {run_id} has {total_size/1024**3:.3f} GB storage, {len(files)} files")
# Returns (for example): Run 26s3qdw9 has 0.239 GB storage, 192 files

total_size_project = wa.get_project_total_storage(entity, project) # will take a minute or so
print(f"Project {project} has {total_size_project/1024**3:.3f} GB total storage")
# Returns (for example): Project allen-10000-56 has 27.547 GB total storage
```
