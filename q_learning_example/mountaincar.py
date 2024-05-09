import gym
import numpy as np
import imageio

LEARNING_RATE = 0.01
DISCOUNT_RATE = 0.9
EPISODES = 10000
SHOW_EPISODES = 200

#Init environment and split observation into discrete parts
env = gym.make("MountainCar-v0", render_mode='rgb_array')
print("Observation High : ", env.observation_space.high)
print("Observation Low : ", env.observation_space.low)
DISCRETE_SIZE = [20] * len(env.observation_space.high)
discrete_gap_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_SIZE

#Init q table
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_SIZE + [env.action_space.n])) # size : [DISCRETE_SIZE, DISCRETE_SIZE, num_actions]

#function to get state represented by integer
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_gap_size
    return tuple(discrete_state.astype(int))

#training based on q - learning 
for ep in range(EPISODES):
    if ep % SHOW_EPISODES == 0:
        render = True
    
    else:
        render = False
    
    init_discrete_state = get_discrete_state(env.reset()[0])
    frames = []
    done = False
    while not done:
        action = np.argmax(q_table[init_discrete_state])
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        new_discrete_state = get_discrete_state(new_state)
        if render:
            frames.append(env.render())
        
        if not done:
            max_new_q = np.max(q_table[new_discrete_state])
            current_q = q_table[init_discrete_state + (action, )]
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_RATE * max_new_q - current_q)
            q_table[init_discrete_state + (action, )] = new_q
        
        elif new_state[0] > env.goal_position:
            print("REACHED !!!!")
            q_table[init_discrete_state + (action, )] = 0

        init_discrete_state = new_discrete_state
    
    if render:
        imageio.mimsave(f'./Results_MountainCar/{ep}.gif', frames, fps=40)


env.close()