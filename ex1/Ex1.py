import gym

env = gym.make('CartPole-v0')
avg_reward = 0
for i_episode in range(10):
    observation = env.reset()
    sum_reward = 0
    for t in range(100):
        env.render()
        print("observation {}: {}".format(t, observation))
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("timestep {}, reward = {}".format(t, reward))
        sum_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
    #avg_reward += sum_reward
    #print("avg_reward", avg_reward)
    print("episode {}, reward = {}".format(i_episode, sum_reward))

env.close()
