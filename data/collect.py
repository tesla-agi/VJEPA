import gymnasium as gym
import numpy as np
import os
from tqdm import tqdm
import ale_py

gym.register_envs(ale_py)

obs_type="grayscale"
sticky=0.0
frameskip=4
frame_convention="post"
def collect_random_rollouts(num_episodes=50, max_steps=1000, save_dir='rollouts'):
    os.makedirs(save_dir,exist_ok=True)
    env=gym.make('ALE/Breakout-v5',
                 obs_type=obs_type,
                 repeat_action_probability=sticky,
                 frameskip=frameskip)
    env.action_space.seed(0)
    for episode in tqdm(range(num_episodes),desc="Episodes"):
        obs,info=env.reset(seed=episode)
        ep_frames=[]
        ep_actions=[]
        ep_lives=[]
        for step in range(max_steps):
            action=env.action_space.sample()
            obs_next,_,terminated,truncated,info=env.step(action)
            done=terminated or truncated
            ep_actions.append(action)
            ep_frames.append(obs_next)
            ep_lives.append(info['lives'])

            if done:
                break

        filename=os.path.join(save_dir,f"episode_{episode:04d}.npz")
        np.savez_compressed(filename,
                            frames=np.array(ep_frames,dtype=np.uint8),
                            actions=np.array(ep_actions,dtype=np.int64),
                            lives=np.array(ep_lives,dtype=np.int64),
                            frame_convention=frame_convention,
                            sticky=sticky,
                            frameskip=frameskip,
                            obs_type=obs_type)


    env.close()
    print(f"Data collection done")
    print(f"Saved {num_episodes} episodes to {save_dir}")


if __name__ == "__main__":
    collect_random_rollouts(num_episodes=300, max_steps=1000, save_dir="rollouts")

