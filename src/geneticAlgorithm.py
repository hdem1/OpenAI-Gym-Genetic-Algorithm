from math import ceil
from tokenize import String
from xmlrpc.client import MAXINT, MININT
import numpy as np
from environmentHandler import environmentHandler
from NeuralNetwork import NeuralNetwork
import random
from os.path import exists, expanduser

class GeneticAlgorithm:

    def __init__(self, env, numGenerations, numChildren, numTestsPerChild = 5, hiddenLayerSizes = [], activationFunctions = [], outputActivationFunction = "S", survivalRate = 0.05, flexibleLayerSizing = True, combinationRatio = 0.3, randomRatio = 0.3):
        self.envHandler = environmentHandler(env)
        self.numGenerations = numGenerations
        self.numChildren = numChildren
        self.flexibleLayerSizing = flexibleLayerSizing
        self.numTestsPerChild = numTestsPerChild
        self.actionRanges = self.envHandler.getActionRanges()
        self.obsRanges = self.envHandler.getObservationRanges()
        self.startingLayerSizes = hiddenLayerSizes
        self.activationFunctions = activationFunctions
        self.activationFunctions.append(outputActivationFunction)
        self.startingLayerSizes.insert(0,len(self.obsRanges))
        self.startingLayerSizes.append(len(self.actionRanges))
        self.survivalRate = survivalRate
        self.numGenerationsDone = 0
        self.combinationRatio = combinationRatio
        self.randomRatio = randomRatio
        self.mutationRatio = 1 - combinationRatio - randomRatio
        self.bestSet = []
        self.folder, self.filename = self.makeNewModelFileName()
        self.modelSaved = False
        for i in range(int(np.ceil(numChildren * survivalRate))):
            #print("HI")
            newNN = NeuralNetwork()
            newNN.makeRandomNeuralNetwork(self.startingLayerSizes, self.activationFunctions)
            self.bestSet.append(newNN)
            #print(self.bestSet[i].getWeights())
        print(self.startingLayerSizes)
    
    def loadModel(self, filename):
        self.modelSaved = False
        self.filename = filename
        file = open(self.folder + filename, "r")
        lines = file.readlines()
        NN = NeuralNetwork()
        NN.makeModelFromStrings(lines)
        self.bestSet[0] = NN
        self.startingLayerSizes = self.bestSet[0].getLayerSizes()
        for i in range(1,len(self.bestSet)):
            randomNN = NeuralNetwork()
            randomNN.makeRandomNeuralNetwork(self.startingLayerSizes, self.activationFunctions)
            self.bestSet[i] = randomNN
    
    def makeNewGeneration(self):
        newGeneration =  []
        mutationsPerPrev = round((self.numChildren - len(self.bestSet)) * self.mutationRatio / len(self.bestSet))
        combinations = round(self.numChildren - len(self.bestSet) * self.combinationRatio)
        randoms = self.numChildren - len(self.bestSet) * (1+mutationsPerPrev) - combinations
        for NN in self.bestSet:
            #print(NN.getWeights())
            newGeneration.append(NN)
            for i in range(mutationsPerPrev):
                #print(NN.getWeights())
                newNN = NeuralNetwork()
                newNN.setWeightsAndBiases(NN.getWeights(), NN.getBiases(), NN.getActFuncts())
                newNN.insertMutations(mutationRate = (i+1)/mutationsPerPrev/4) #1 -> 1/2 -> 1/4
                #print(newNN.getWeights())
                newGeneration.append(newNN)
            #print("--------")
        for i in range(combinations):
            index1 = int(np.floor(random.random() * len(self.bestSet)))
            index2 =int(np.floor(random.random() * len(self.bestSet)))
            while index2 == index1 and len(self.bestSet) != 1:
                index2 = int(np.floor(random.random() * len(self.bestSet)))
            newNN = NeuralNetwork()
            newNN.makeCombination(self.startingLayerSizes, [self.bestSet[index1], self.bestSet[index2]])
            newNN.insertMutations(mutationRate = 0.05)
            newGeneration.append(newNN)
        for i in range(randoms):
            newNN = NeuralNetwork()
            newNN.makeRandomNeuralNetwork(self.startingLayerSizes, self.activationFunctions)
            newGeneration.append(newNN)
        return newGeneration
    
    def simulateGeneration(self, printProgress = True, modifyReward=False):
        if printProgress:
            print("Progress: [", end ="", flush = True)
            lastPrint = 0
        newGeneration = self.makeNewGeneration()
        rewards = []
        for i in range(self.numChildren):
            if printProgress and i - lastPrint >= 0.1 * self.numChildren:
                lastPrint = i
                print("*",end = "", flush = True)
            rewards.append(self.envHandler.runMultipleSimulations(self.numTestsPerChild, newGeneration[i], modifyReward=modifyReward))#, displaying = True))
            #print(rewards[i])
        if printProgress:
            print("]")

        bestRewards = []
        for i in range(int(np.ceil((self.numChildren * self.survivalRate)))):
            maxReward = MININT
            maxIndex = 0
            for j in range(self.numChildren-i):
                if rewards[j] > maxReward:
                    maxReward = rewards[j]
                    maxIndex = j
            self.bestSet[i] = newGeneration.pop(maxIndex)
            bestRewards.append(rewards.pop(maxIndex))
        if printProgress:
            print("Best Rewards =",bestRewards)
        
    def train(self, printProgress = True, displayBest = True, numDisplayIterations = 2, saveOldModel = True, savePerGen = True, endTests = 100, modifyReward = False):
        if saveOldModel == True and self.modelSaved:
            self.modelSaved = False
            self.folder, self.filename = self.makeNewModelFileName()
        printing = printProgress
        for i in range(self.numGenerations):
            if printing:
                print("\nGeneration ",(i+1),":", sep ="")
            self.simulateGeneration(printProgress = printing, modifyReward = modifyReward)
            if savePerGen:
                self.saveBestModel(printInfo= False)
            if displayBest:
                self.envHandler.runMultipleSimulations(numDisplayIterations, self.bestSet[0], displaying=True)
        #Resorting the final set with more data:
        print("\nSorting Best Networks...")
        rewards = []
        for NN in self.bestSet:
            rewards.append(self.envHandler.runMultipleSimulations(endTests, NN))
        newBestSet = []
        num = len(self.bestSet)
        for i in range(num):
            maxReward = MININT
            maxIndex = 0
            for j in range(len(self.bestSet)):
                if rewards[j] > maxReward:
                    maxReward = rewards[j]
                    maxIndex = j
            newBestSet.append(self.bestSet.pop(maxIndex))
            rewards.pop(maxIndex)
        self.bestSet = newBestSet

    def testBest(self, iterations, saving = True):
        avg_reward = 0
        print("\nTesting for", iterations, "iterations...")
        avg_reward, avg_iterations = self.envHandler.runMultipleSimulations(iterations, self.bestSet[0], returnIterations = True)
        print("\nTest Results:")
        indent = "   "
        print(indent, "- Average Reward = ", avg_reward)
        if saving:
            self.savePerformance(avg_reward, avg_iterations, printInfo = False)

    def displayBest(self, iterations = -1, printRewards = False):
        i = 0
        while i < iterations or iterations == -1:
            avg_reward, avg_iterations = self.envHandler.runSimulation(self.bestSet[0],displaying = True)
            if printRewards:
                print("Reward = ", avg_reward, "; Iterations = ", avg_iterations, sep = "")
            i += 1

    def makeNewModelFileName(self):
        folder = expanduser("~/Documents/Random Coding Projects/MachineLearningExperiments/OpenAI-Gym-Genetic-Algorithm/Saved Models/")
        filename = self.envHandler.getEnvironmentName()
        filename = filename + "_gens-" + str(self.numGenerations)
        filename = filename + "_children-"+str(self.numChildren) 
        filename = filename + "_layers-" + str(self.startingLayerSizes)
        filename = filename + "_networkTests-"+str(self.numTestsPerChild)
        if exists(folder + filename + ".txt"):
            value = 1
            while (exists(folder +filename + "_"+str(value))):
                value+=1
            filename = filename + "_" + str(value)
        filename = filename +".txt"
        return folder,filename
    
    def saveBestModel(self, printInfo = True):
        file = open(self.folder+self.filename, "w")
        if printInfo:
            print("Filename =", self.filename)

        #Writing data:
        if printInfo:
            print("Saving neural network...")
        file.write(self.bestSet[0].getModelString())

        file.close()

    def savePerformance(self, reward, iterations, printInfo = True):
        #All following lines = training performances
        if printInfo:   
            print("Saving performance statistics...")
        file = open(self.folder + self.filename, "a")
        lastline = []
        lastline.append(str(reward)+",")
        lastline.append(str(iterations)+"\n")
        file.writelines(lastline)
        file.close()
    
    def startNewFile(self):
        self.folder, self.filename = self.makeNewModelFileName()
        self.modelSaved = False
    
    def close(self):
        self.envHandler.closeEnvironment()