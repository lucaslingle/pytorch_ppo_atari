import torch as tc
import os


def _format_name(kind, steps):
    filename = f"{kind}_{steps}.pth"
    return filename


def _parse_name(filename):
    kind, steps = filename.split(".")[0].split("_")
    steps = int(steps)
    return {
        "kind": kind,
        "steps": steps
    }


def _latest_n_checkpoint_steps(base_path, n=5):
    steps = set(map(lambda x: _parse_name(x)['steps'], os.listdir(base_path)))
    latest_steps = sorted(steps)
    latest_n = latest_steps[-n:]
    return latest_n


def _latest_step(base_path):
    return _latest_n_checkpoint_steps(base_path, n=1)[-1]


def save_checkpoint(checkpoint_dir, model_name, agent, steps):
    """
    Saves a checkpoint of the latest model, optimizer, scheduler state.
    Also tidies up checkpoint_dir/model_name/ by keeping only last 5 checkpoints.

    :param checkpoint_dir: str checkpoint dir for checkpointing.
    :param model_name: str model name for checkpointing.
    :param agent: agent_utils.Agent encapsulating model, optimizer, scheduler, comm.
    :param steps: int env steps experienced by the agent to the checkpoint.
    :return:
    """
    base_path = os.path.join(checkpoint_dir, model_name)
    os.makedirs(base_path, exist_ok=True)

    # save everything
    tc.save(agent.model.state_dict(),
            os.path.join(base_path, _format_name('model', steps)))
    tc.save(agent.optimizer.state_dict(),
            os.path.join(base_path, _format_name('optimizer', steps)))
    tc.save(agent.scheduler.state_dict(),
            os.path.join(base_path, _format_name('scheduler', steps)))

    # keep only last n checkpoints
    latest_n_steps = _latest_n_checkpoint_steps(base_path, n=5)
    for file in os.listdir(base_path):
        if _parse_name(file)['steps'] not in latest_n_steps:
            os.remove(os.path.join(base_path, file))


def maybe_load_checkpoint(checkpoint_dir, model_name, agent, steps=None):
    """
    Tries to load a checkpoint from checkpoint_dir/model_name/.
    If there isn't one, it fails gracefully, allowing the script to proceed
    from a newly initialized model.

    :param checkpoint_dir: str checkpoint dir for checkpointing.
    :param model_name: str model name for checkpointing.
    :param agent: agent_utils.Agent encapsulating model, optimizer, scheduler, comm.
    :param steps: int env steps for the checkpoint to locate. if none, use latest.
    :return: int number of env steps experienced by loaded checkpoint.
    """
    base_path = os.path.join(checkpoint_dir, model_name)
    try:
        if steps is None:
            steps = _latest_step(base_path)

        agent.model.load_state_dict(
            tc.load(os.path.join(base_path, _format_name('model', steps))))
        agent.optimizer.load_state_dict(
            tc.load(os.path.join(base_path, _format_name('optimizer', steps))))
        agent.scheduler.load_state_dict(
            tc.load(os.path.join(base_path, _format_name('scheduler', steps))))

        print(f"Successfully loaded checkpoint from {base_path}, with step {steps}.")
    except Exception:
        print(f"Bad checkpoint or none at {base_path}. Continuing from scratch.")
        steps = 0

    return steps
