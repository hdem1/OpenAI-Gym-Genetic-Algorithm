from os import environ
import gym
import numpy as np
from NeuralNetwork import NeuralNetwork

class environmentHandler:

    def __init__(self, environment):
        self.env = gym.make(environment)
        self.environmentName = environment
        self.observationSpace = self.env.observation_space
        self.actionSpace = self.env.action_space
        #print(type(self.actionSpace)== gym.spaces.discrete.Discrete)
        #print(self.actionSpace.shape)
    
    def closeEnvironment(self):
        self.env.close()

    def getActionRanges(self):
        index = 0
        output = []
        if type(self.actionSpace) == gym.spaces.box.Box:
            for i in range(self.actionSpace.shape[0]):
                output.append([self.actionSpace.low[index], self.actionSpace.high[index]])
        if type(self.actionSpace) == gym.spaces.discrete.Discrete:
            return [[self.actionSpace.start, self.actionSpace.start + self.actionSpace.n - 1]]
        return output
    
    def getObservationRanges(self):
        index = 0
        output = []
        for i in range(self.observationSpace.shape[0]):
            output.append([self.observationSpace.low[index], self.observationSpace.high[index]])
        return output

    def runSimulation(self, neuralNetwork:NeuralNetwork, maxIterations = -1, displaying = False, modifyReward = False):
        done = False
        obs = self.env.reset()
        
        obsArray = np.array(obs, dtype = object).reshape(len(obs),1)
        #while (obsArray[1])
        iterations = 0
        totalReward = 0
        while not done and (maxIterations == -1 or iterations <= maxIterations):
            output = neuralNetwork.getOutput(obsArray)
            #print(output)
            action = []
            actionRanges = self.getActionRanges()
            for i in range(len(output)):
                action.append(output[i] * (actionRanges[i][1]-actionRanges[i][0]) + actionRanges[i][0])
            if type(self.actionSpace) == gym.spaces.discrete.Discrete:
                action = int(np.round(action[0]))
            #print(output, "->", action)
            # if obsArray[1][0] < 0.5:
            #     action = [obsArray[2][0]/8 - obsArray[1][0]]
            # else:
            #     action = [-0.25 * obsArray[2][0] +  obsArray[0][0]]
            #     print(action)
            # action[0] = min(max(action[0], -2), 2)
            obs, reward, done, info = self.env.step(action)
            obsArray = np.array(obs, dtype = object).reshape(len(obs), 1)
            #print(obs, " --> ", obsArray)
            #print(obsArray.reshape(1, len(obs)), " -> ", ((1 - abs(obsArray[2][0])/8)**2) * obsArray[1][0] * 100)
            totalReward += reward 
            if modifyReward and self.environmentName == "Pendulum-v1":
                totalReward += ((1 - abs(obsArray[2][0])/8)**2) * obsArray[0][0] * 10
            if modifyReward and self.environmentName == "LunarLander-v2":
                totalReward += 3 - (obsArray[1][0] + 1.5)
            if displaying:
                self.env.render()
            iterations += 1
            # if self.environmentName == "BipedalWalker-v3":
            #     if (not done and 10 * totalReward + 100 < iterations) or (done and iterations >= 1599):
            #         #print(iterations)
            #         #totalReward -= 100
            #         break
        return totalReward, iterations
    
    def runMultipleSimulations(self, num_tests:int, neuralNetwork:NeuralNetwork, maxIterations = -1, displaying = False, successChecking = False, successRewardThreshold = 0, successIterationThreshold = -1, returnIterations = False, modifyReward = False):
        successRate = 0
        avg_reward = 0
        avg_iterations = 0
        for t in range(num_tests):
            reward, iterations = self.runSimulation(neuralNetwork, maxIterations = maxIterations, displaying = displaying, modifyReward = modifyReward)
            avg_reward += reward
            avg_iterations += iterations
            if successChecking:
                if successIterationThreshold >= 0 and iterations >= successIterationThreshold:
                    successRate +=1
                elif successIterationThreshold == -1 and reward >= successRewardThreshold:
                    successRate +=1
        avg_reward /= num_tests
        avg_iterations /= num_tests
        if successChecking and returnIterations:
            successRate /= num_tests
            return avg_reward, avg_iterations, successRate
        elif successChecking:
            successRate /= num_tests
            return avg_reward, successRate
        elif returnIterations:
            return avg_reward, avg_iterations
        return avg_reward
            
    def getEnvironmentName(self):
        return self.environmentName