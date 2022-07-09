import numpy as np
import random
import copy

#First Initializer taken from Sebastian Lague Youtube Channel - https://www.youtube.com/watch?v=d9hLNUzLBYI&list=RDCMUCmtyQOKKmrMVaKuRXz02jbQ 

class NeuralNetwork:
    
    def __init__(self):
        self.weights = []
        self.biases = []
        self.layers = []
        self.actFuncts = []
    
    def makeModelFromStrings(self, lines):
        firstLine = lines[0].split(",")
        numLayers = int(firstLine[0])
        self.layers = []
        for i in range(numLayers):
            self.layers.append(int(firstLine[i+1]))
        weight_shapes = [(a,b) for a,b in zip(self.layers[1:], self.layers[:-1])]
        self.weights = [np.zeros(s) for s in weight_shapes]
        self.biases = [np.zeros((s,1)) for s in self.layers[1:]]
        self.actFuncts =[[""] * s for s in self.layers[1:]]

        layer = 0
        node = 0
        numNodes = sum(self.layers[1:])
        for line in lines[1:(numNodes+1)]:
            nums = line.split(",")
            for i in range(self.layers[layer]):
                self.weights[layer][node][i] = float(nums[i])
            self.biases[layer][node][0] = float(nums[-2])
            self.actFuncts[layer][node] = nums[-1][:-1]

            node += 1
            if node >= self.layers[layer+1]:
                node = 0
                layer += 1
        #print(self.getModelString())
        
    def makeRandomNeuralNetwork(self, layerSizes, layerFuncs):
        weight_shapes = [(a,b) for a,b in zip(layerSizes[1:], layerSizes[:-1])]
        self.weights = [np.random.standard_normal(s)/s[1]**0.5 for s in weight_shapes] 
        self.biases = [np.random.standard_normal((s,1))/s**0.5 for s in layerSizes[1:]]
        self.actFuncts =[[layerFuncs[i]] * layerSizes[i+1] for i in range(len(layerFuncs))]
        self.layers = layerSizes
        #print(self.layers)

    def setWeightsAndBiases(self, weights, biases, actFuncts):
        self.weights = weights
        self.biases = biases
        self.layers = [len(self.weights[0][0])]
        for i in self.weights:
            self.layers.append(len(i))
        self.actFuncts = actFuncts
    
    def makeCombination(self, layerSizes, otherNetworks):
        self.layers = layerSizes
        weight_shapes = [(a,b) for a,b in zip(layerSizes[1:], layerSizes[:-1])]
        self.weights = [np.zeros(s) for s in weight_shapes]
        self.biases = [np.zeros((s,1)) for s in layerSizes[1:]]
        self.actFuncts =[[""] * s for s in layerSizes[1:]]
        for layer in range(len(layerSizes)-1):
            for node in range(layerSizes[layer+1]):
                randIndex = int(np.floor(random.random() * len(otherNetworks))) # Make this based on the previous values mb?
                w, b, f = otherNetworks[randIndex].getWeightsAndBias(layer, node)
                self.weights[layer][node] = w
                self.biases[layer][node] = b 
                self.actFuncts[layer][node] = f
    
    def getNetwork(self):
        return copy.deepcopy(self.weights), copy.deepcopy(self.biases)
    
    def getWeightsAndBias(self, layer, index):
        return copy.deepcopy(self.weights[layer][index]), copy.deepcopy(self.biases[layer][index]), copy.deepcopy(self.actFuncts[layer][index])
    
    def getWeights(self):
        return copy.deepcopy(self.weights)
    
    def getBiases(self):
        return copy.deepcopy(self.biases)
    
    def getActFuncts(self):
        return copy.deepcopy(self.actFuncts)

    def getLayerSizes(self):
        return copy.deepcopy(self.layers)

    def getOutput(self, input): 
        lastLayer = np.array(input).ravel()
        for layer in range(len(self.weights)):
            newLayer = []
            for node in range(len(self.weights[layer])):
                val = self.biases[layer][node][0]
                for prevNode in range(len(self.weights[layer][node])):
                    val += self.weights[layer][node][prevNode] * lastLayer[prevNode]
                newLayer.append(self.activation(val, self.actFuncts[layer][node]))
            lastLayer = newLayer
        return lastLayer

        # lastLayer = input
        # for w, b in zip(self.weights, self.biases):
        #     print(lastLayer)
        #     #print("lastLayer =", lastLayer)
        #     #print("w =", w)
        #     weightSum = np.matmul(w,lastLayer)
        #     #print("weightSum =", weightSum)
        #     #print("b =",b)
        #     #print("weightSum+b =", weightSum+b)
        #     #print("numpy sum =", np.add(weightSum, b))
        #     lastLayer = self.activation((np.add(weightSum,b)))
        # lastLayer = lastLayer.ravel()
        # #print("Final output =", lastLayer)
        # return lastLayer
    
    def activation(self, value, function):
        #Sigmoid:
        output = 0
        #sigmoid:
        if function == "S": 
            output = 1/(1+np.exp(-value))
        #Leaky Relu:
        if function == "LR": 
            alpha = 0.01
            output = max(alpha * value, value) #[max(alpha * x, x) for x in floatLayer]
        #Normal Relu:
        if function == "R":
            output = max(0, value)
        
        return output
    
    def insertMutations(self, mutationRate = 0.1, mutationMagnitude = 2):
        for layer in range(len(self.weights)):
            for node in range(len(self.weights[layer])):
                for weight in range(len(self.weights[layer][node])):
                    if random.random() < mutationRate:
                        #self.weights[layer][node][weight] += random.gauss(0,mutationMagnitude)
                        self.weights[layer][node][weight] += random.gauss(0,mutationMagnitude)
                if random.random() < mutationRate:
                    #self.biases[layer][node] += random.gauss(0,mutationMagnitude)
                    self.biases[layer][node] += random.gauss(0,mutationMagnitude)
    
    def getModelString(self):
        output = ""
        #General network shape:
        output = output + str(len(self.layers))+","
        for i in range(len(self.layers)-1):
            output = output + str(self.layers[i])+","
        output = output + str(self.layers[-1])+"\n"
        
        #Weights and biases - each row represents a node
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    output = output + str(self.weights[i][j][k]) + ","
                output = output + str(self.biases[i][j][0]) + "," + self.actFuncts[i][j] + "\n"
        
        return output


    
