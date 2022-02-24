import time
import gym
import numpy as np
from numpy.core.defchararray import count


class Network:

    def __init__(self,env, nhiddens):


        self.cumreward =[]
        self.pvariance = 0.1     # variance of initial parameters
        self.ppvariance = 0.02   # variance of perturbations
        self.nhiddens = nhiddens       # number of internal neurons
        # the number of inputs and outputs depends on the problem
        # we assume that observations consist of vectors of continuous value
        # and that actions can be vectors of continuous values or discrete actions
        self.ninputs = env.observation_space.shape[0]
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            self.noutputs = env.action_space.shape[0]
        else:
            self.noutputs = env.action_space.n
        
        # initialize the training parameters randomly by using a gaussian
        # distribution with average 0.0 and variance 0.1
        # biases (thresholds) are initialized to 0.0
        self.W1 = np.random.randn(self.nhiddens,self.ninputs) * self.pvariance      # first connection layer
        self.W2 = np.random.randn(self.noutputs, self.nhiddens) * self.pvariance    # second connection layer
        self.b1 = np.zeros(shape=(self.nhiddens, 1))                      # bias internal neurons
        self.b2 = np.zeros(shape=(self.noutputs, 1))                      # bias motor neurons
        self.env_w = [self.W1 , self.W2 ,self.b1 ,self.b2]

    def update(self,observation):
        
        # change chape to be able to multiply the matrix between observation and connection and weights
        # convert the observation array into a matrix with 1 column and ninputs rows
        observation.resize(self.ninputs,1)
        # compute the netinput of the first layer of neurons
        self.Z1 = np.dot(self.W1, observation) + self.b1
        # compute the activation of the first layer of neurons with the tanh function
        self.A1 = np.tanh(self.Z1)
        # compute the netinput of the second layer of neurons
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        # compute the activation of the second layer of neurons with the tanh function
        self.A2 = np.tanh(self.Z2)
        # if the action is discrete
        #  select the action that corresponds to the most activated unit
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            action = self.A2 # if the action is integer [0 , 1]
        else:
            action = np.argmax(self.A2)  #

        return(action)
    def compute_parameters(self):
        all_param = np.array(self.env_w)
        a =[]
        # for idx,i in enumerate(self.env_param):
        for i in all_param:
            # print(np.shape(i))
            size = 1
            for dim in np.shape(i): 
                size *= dim
                # print(size)
            a.append(size)
            nparameters = sum(a)
        print("nparams : " , nparameters)
        # print("all_param : " , i.flatten())
        print(a)
        nparameters = len(all_param)
        
        return nparameters


    def set_genotype(self, genotype):
        #  setting the values of the genotype and adding the weights and biasses values.
        self.env_w = genotype;

    def evaluate(self, nepisodes):
        cumreward = 0
        # self.env.render(mode = 'rgb_array')
        for e in range(nepisodes):
            observation = env.reset()
            done = False
            while not done :
                action = network.update(observation)
                observation, reward, done, info = env.step(action)
                cumreward += reward


        return cumreward/nepisodes

    def render(self, nepisodes):

        for e in range(nepisodes):
            self.cumreward =[]
            observation = env.reset()
            done = False
            while not done :
                env.render()
                action = network.update(observation)
                observation, reward, done, info = env.step(action)
                self.cumreward.append(reward)
                time.sleep(0.05)
                print("Cumulatitive reward: ", sum(self.cumreward))
        show_video()
        env.close()
        return reward

# env = gym.make("CartPole-v1")
# env = gym.make("MountainCarContinuous-v0")
# env = gym.make("MountainCar-v0")
#env=wrap_env(gym.make('CartPole-v0'))
env= wrap_env(gym.make('Acrobot-v1'))

network= Network(env, 5)

popsize=10
variance=0.1
pertrub_variance=0.02
ngenerations=100
episodes=3

nparameters = network.compute_parameters()
population =  np.random.randn(popsize, nparameters)*variance

for g in range(ngenerations):
  fitness=[]
  for i in range(popsize):
    network.set_genotype(population[i])
    fit=network.evaluate(episodes)
    fitness.append(fit)
  # replacing the worst genotype with perturbed versions of the best genotypes
  indexbest=np.argsort(fitness)
  for i in range(int(popsize/2)):
    population[indexbest[i+5]] = population[indexbest[i]]+np.random.rand(nparameters)*pertrub_variance

network.render(episodes)