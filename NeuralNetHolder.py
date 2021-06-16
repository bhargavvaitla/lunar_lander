import numpy as np
import math
import pandas as pd 
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


lamda = 0.7
learning_rate = 0.1
momemtam_rate = 0.05

class Neuron:
    def __init__(self,noInputs, isHidNeuron):
        self.weights =np.random.random(noInputs+1)
        self.oldWeights =[0.0]*(noInputs+1)
        self.output = 0.0
        self.deltaWeights =[0.0]*(noInputs+1)
        self.hiddenNeuron = isHidNeuron
    
    def sigmoidActivation(self,output):
        return 1.0/(1+ np.exp(-1.0*lamda*output))
    
    def neuronActivation(self, inputs):
        neuron_output = 0
        neuron_weights = self.weights
        for i in range(len(inputs)):
            neuron_output = neuron_output + inputs[i]*neuron_weights[i]
        neuron_output = neuron_output + neuron_weights[-1]
        
        if self.hiddenNeuron == True:
            self.output = self.sigmoidActivation(neuron_output)
        else:
            self.output = neuron_output
            
        return self.output
    
    def outputGradient(self, value):
        return value - self.output 
    
    def updateWeights(self,gradeint, inputs):
        weights = self.weights
        deltaWeights = self.deltaWeights
        updated_weights = []
        delta_weights   = []
        for i in range(len(weights)):
            del_weight = (learning_rate * gradeint * inputs[i]) + (momemtam_rate * deltaWeights[i])
            new_weight = weights[i]+ del_weight
            updated_weights.append(new_weight)
            delta_weights.append(del_weight)
        self.oldWeights = weights
        self.weights = updated_weights
        self.deltaWeights = delta_weights
    
    def getHiddenGradient(self,gardients,outputLayer,loc):
        gardient = 0
        neu_output = self.output
        for i in range(len(outputLayer)):
            weights = outputLayer[i].oldWeights
            gardient = gardient + gardients[i]*weights[loc]
    
        return lamda * neu_output * (1-neu_output) * gardient
        


class Network:
    def __init__(self,input_size,hidden_size,output_size):
            net = []
            hidden_layer = []
            output_layer =[]
            for i in range(hidden_size):
                hidden_layer.append(Neuron(input_size,True))
            for j in range(output_size):
                output_layer.append(Neuron(hidden_size,False))
        
            net.append(hidden_layer)
            net.append(output_layer)
            self.network = net
    
    def forwardPropogation(self,inputs):
        network = self.network
        layer_inputs = inputs 
        for layer in network:
            layer_outputs = []
            for neuron in layer:
                act_val = neuron.neuronActivation(layer_inputs)
                layer_outputs.append(act_val)
            layer_inputs = layer_outputs
        return layer_inputs
    
    
    def getHiddenLayerOutputs(self):
        hidden_layer = self.network[0]
        hidden_layer_outputs = []
        for neu in hidden_layer:
            hidden_layer_outputs.append(neu.output)
        hidden_layer_outputs.append(1)
        return hidden_layer_outputs
    
    def backPropagation(self,inputs,outputs, expectedOutputs):
        network = self.network
        hidden_layer = network[0]
        output_layer = network[1]
        inputs.append(1)

        hidden_layer_outputs = self.getHiddenLayerOutputs()
        output_layer_grad =[]
        
        for i in range(len(outputs)):
            out_neu =  output_layer[i]
            out_gradient= out_neu.outputGradient(expectedOutputs[i])
            output_layer_grad.append(out_gradient)
            out_neu.updateWeights(out_gradient,hidden_layer_outputs)
    
        for j in range(len(hidden_layer)):
            hid_neu = hidden_layer[j]
            hid_gradient = hid_neu.getHiddenGradient(output_layer_grad,output_layer,j)
            hid_neu.updateWeights(hid_gradient,inputs)
        
        

        
        
    
        
        
        
        
def getErorr(outputs,expected_outputs):
    erorr = 0
    for i in range(len(outputs)):
        diff = (outputs[i]-expected_outputs[i]) ** 2
        erorr = erorr + diff
    return erorr/2
    
def main():
    hidden_neurons = 4
    network = Network(2,hidden_neurons,2)
    data = pd.read_csv("prabath_sample_add.csv") 
    data_1 = shuffle(data)

    data_range = data_1[0:30000]
    
    inp_1_minMax = [data_range['input1'].min(),data_range['input1'].max()]
    inp_2_minMax = [data_range['input2'].min(),data_range['input2'].max()]
    
    print(inp_1_minMax)
    print(inp_2_minMax)
    
    test_range = data_1[41000:52000]
    norm_data = normaliseData(data_range)
    norm_test = normaliseData(test_range)
    input_size = len(norm_data.index)
    test_size = len(norm_test.index)
    
    rms_prev = 0.0
    diff_threshold = 1/100000
    diff_count = 0
    error_inc_count = 0
    
    error_list = []
    test_error_list = []
    for i in range(300):
        epoch_erorr = 0.0
        test_erorr = 0.0
        for row in norm_data.values.tolist():
            inputs = row[0:2]
            expected_outputs = row[2:4]
            outputs = network.forwardPropogation(inputs)
            network.backPropagation(inputs,outputs,expected_outputs)
            erorr = getErorr(outputs,expected_outputs)
            epoch_erorr = epoch_erorr + erorr
        for row in norm_test.values.tolist():
            inputs = row[0:2]
            expected_outputs = row[2:4]
            outputs = network.forwardPropogation(inputs)
            erorr = getErorr(outputs,expected_outputs)
            test_erorr = test_erorr + erorr
            
        rms_training = math.sqrt(epoch_erorr/input_size)
        rms_validation =math.sqrt( test_erorr/test_size)
        print(str(i)+ ' '+ str(rms_training) + ' '+ str(rms_validation))
        
        if rms_validation > rms_prev:
            error_inc_count = error_inc_count+1
        else:
            error_inc_count =0
        
        val_diff = abs(rms_validation-rms_prev)
     
        if val_diff < diff_threshold:
            diff_count = diff_count+1
        else:
            diff_count=0
        
        if error_inc_count == 10 or  diff_count ==10:
            break
        error_list.append(rms_training)
        test_error_list.append(rms_validation)
        rms_prev = rms_validation
        #if i%5 == 0.0:
        #    printWeights(network)
    
    
    plt.plot(error_list)
    plt.plot(test_error_list)
    
    plt.xlabel('Epoch')
    plt.ylabel('Rms')
    plt.legend(["training", "validation"])
    plt.title("neurons: "+ str(hidden_neurons)+" learning: "+ str(learning_rate)+  " momentum: "+str(momemtam_rate))
    
    
    

def printWeights(network):
    net = network.network
    hid_layer = net[0]
    outer_layer = net[1]
    
    for i in range(len(hid_layer)):
        weights = hid_layer[i].weights
        print('hidden neuron'+' '+str(i)+ ' '+str(weights))
    
    for i in range(len(outer_layer)):
        weights = outer_layer[i].weights
        print('outer neuron'+' '+str(i)+ ' '+str(weights))

        
        
class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        net = Network(2,4,2)
        hidden_layer = net.network[0]
        outer_layer = net.network[1]
        hidden_layer[0].weights = [10.351190402605896, -24.48992026704253, -4.73873153219747]
        hidden_layer[1].weights = [102.52421716727693, 1.1939277543535016, -55.789744648782005]
        hidden_layer[2].weights = [-9.194983531553541, -19.568235273146712, 5.5242563047790005]
        hidden_layer[3].weights = [-86.5790259699908, 8.548077108987219, 40.021170073805365]
        
        outer_layer[0].weights =[2.134211776996221, -2.2634614923378167, 3.2582213576180825, -2.9343791010468503, 1.9953497806446239]
        outer_layer[1].weights = [5.344864791410035, 0.5324669265029361, -5.120143541415624, -0.7808481968318868, -0.03297394683760349]
				
        self.network = net        
        
        

    
    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        input1 = input_row.split(',')
        inp = normaliseData(input1)
        network = self.network
        output = network.forwardPropogation(inp)
        return output


def normaliseData(data):
    inp1 = (float(data[0])+788.6065317996778)/(726.7546455212716 + 788.6065317996778)
    inp2 = (float(data[1])-65.02018975933845)/(861.319194035123-65.02018975933845 )
    return [inp1,inp2]

