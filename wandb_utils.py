import wandb
import re
import os 
import fnmatch

def run_meta_from_url(url):
    """
    Example: 
        url="https://wandb.ai/jmhb0/allen-10000-56/runs/26s3qdw9?workspace=user-jmhb0")
        entity, project, run_id = run_meta_from_url(url)
    """
    regex = r"wandb\.ai\/([^\/]*)\/([^\/]*)\/([^\/]*)\/([a-z0-9]*)"
    matches = re.finditer(regex, url, re.MULTILINE)
    entity, project, _, run_id = list(matches)[0].groups()
    return entity, project, run_id

def get_run(entity, project, run_id):
    api=wandb.Api()
    run=api.run(os.path.join(entity,project,run_id))
    return run

def get_run_files(run, print_sz=1):
    """ """
    files = run.files()
    files=list(files)
    total_size = 0
    for f in files:
        name, size, = f.name, f.size
        if print_sz: print(f"\t{size/1024**3:.4f}GB {name}")
        total_size+=size
    if print_sz: print(f"\nTotal size {total_size/1024**3:.4f} GB")

    return files, total_size

def delete_model_files_except_one(files, keep_model= 'model_epoch_108.pt'):
    """ 
    Args:
        Files: files iterator (output of get_run_files).
    """
    names = [f.name for f in list(files)]
    saved_model_fnames = fnmatch.filter(names, "model_epoch_*.pt")
    assert keep_model in saved_model_fnames
    saved_model_fnames.remove(keep_model)
    delete_files = saved_model_fnames
    for f in files:
        if f.name in delete_files:
            f.delete()

def get_project_total_storage(entity, project):
    api=wandb.Api()
    path = os.path.join(entity, project)
    project_runs = api.runs(path, filters={})

    total_size_project=0
    for run in project_runs: 
        files, total_size = get_run_files(run, print_sz=0)
        total_size_project+=total_size
    total_size_project
    return total_size_project

def delete_model_files_except_one_whole_project(entity, project, keep_model= 'model_epoch_108.pt'):
    """ Careful - deleting lots of files  """
    project_runs = api.runs(os.path.join(entity, project), filters={})
    for run in project_runs:
        files, total_size = wandb_utils.get_run_files(run, print_sz=0)
        print(f"Before delete: {total_size/1024**3:.3f} GB {run.name}")
        wandb_utils.delete_model_files_except_one(files, keep_model=keep_model)

        files, total_size = wandb_utils.get_run_files(run, print_sz=0)
        print(f"After delete: {total_size/1024**3:.3f} GB {run.name}")
        print("\n")

def wandb_load_run(run_id, project="allen-10000-56", user="jmhb0", print_cfg=1,
                  ):
    api=wandb.Api()
    wandb.login()
    path=os.path.join(user, project,run_id)
    print(path)
    run=api.run(path=path)
    if print_cfg:
        name = run.config['model']['name']
        msg = f"\tname: {run.config['model']['name']}\n"+\
              f"\tzdim: {run.config['model']['zdim']}\n"+\
              f"\tbeta: {run.config['loss']['beta']}\n"+\
              f"\tmodel_name {run.config['model']['name']}\n"
        if name=="VadeO2":
            msg+=f"\tn_clust: {run.config['model']['n_clusters']}\n"
        print(msg)
    return run

def wandb_restore_run(run, model, fname_model_weights, tmp_path="./tmp"):
    """
    Args:
        model (torch.nn.Module): torch model to update. 
    """
    run_path="/".join(run.path)
    restore_state=wandb.restore(fname_model_weights, run_path=run_path, 
                                root=tmp_path, replace=True)
    # get the dictionary
    restore_state=torch.load(restore_state.name)
    # update the model weights
    model.load_state_dict(restore_state['model_state_dict'])

    return model
