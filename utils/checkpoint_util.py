import torch as tc
import os


def save_checkpoint(checkpoint_dir, model_name, agent):
    dir_path = os.path.join(checkpoint_dir, model_name)
    os.makedirs(dir_path, exist_ok=True)
    tc.save(agent.model.state_dict(), os.path.join(dir_path, 'model.pth'))
    tc.save(agent.optimizer.state_dict(), os.path.join(dir_path, 'optimizer.pth'))
    tc.save(agent.scheduler.state_dict(), os.path.join(dir_path, 'scheduler.pth'))


def maybe_load_checkpoint(checkpoint_dir, model_name, agent):
    dir_path = os.path.join(checkpoint_dir, model_name)
    try:
        agent.model.load_state_dict(tc.load(os.path.join(dir_path, 'model.pth')))
        agent.optimizer.load_state_dict(tc.load(os.path.join(dir_path, 'optimizer.pth')))
        agent.scheduler.load_state_dict(tc.load(os.path.join(dir_path, 'scheduler.pth')))
        print(f"Successfully loaded checkpoint from {dir_path}")
    except Exception:
        print(f"Bad checkpoint or none at {dir_path}. Continuing training from scratch.")
