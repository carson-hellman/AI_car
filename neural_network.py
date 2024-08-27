import numpy as np
import pandas as pd

class DataPoint:
    def __init__(self, inputs):
        self.inputs = inputs
        self.expected_outputs = self.create_one_hot(inputs)


    def create_one_hot(self, inputs):
        # takes inputs (in this case radar distances) and generates one hot vector. To my understanding it functions as the expected/correct output
        left = inputs[0]
        front = inputs[1]
        right = inputs[2]

        if left < right and left < front:
            # turn right
            return [1, 0, 0] # originially [1, 0, 0]
        elif right < left and right < front:
            # turn left
            return [0, 0, 1] # originally [0, 0, 1]
        else:
            # continue straight
            return [0, 1, 0]
        

class Layer:
    def __init__(self, num_nodes_in, num_nodes_out):
        self.nodes_in = num_nodes_in
        self.nodes_out = num_nodes_out

        self.cost_gradient_w = np.zeros((num_nodes_in, num_nodes_out))
        self.weights = np.random.randn(num_nodes_in, num_nodes_out)

        self.cost_gradient_b = np.zeros(num_nodes_out)
        self.biases = np.zeros(num_nodes_out)

        self.weighted_inputs = None
        self.outputs = None
        self.inputs = None


    def apply_gradients(self, learn_rate):
        for node_out in range(0, self.nodes_out):
            self.biases[node_out] -= self.cost_gradient_b[node_out] * learn_rate

            for node_in in range(0, self.nodes_in):
                self.weights[node_in, node_out] -= self.cost_gradient_w[node_in, node_out] * learn_rate


    def activation_function(self, weighted_input):
        # sigmoid function
        return 1.0 / (1 + np.exp(-weighted_input))
        
        # ReLU
        # return np.maximum(0, weighted_input)

        # leaky ReLU
        # return np.maximum(0.01 * weighted_input, weighted_input)
    

    def activation_function_derivative(self, weighted_input):
        # sigmoid
        activation = self.activation_function(weighted_input)
        return activation * (1 - activation)

        #ReLU
        # return 1 if weighted_input > 0 else 0

        # leaky ReLU
        # return 1 if weighted_input > 0 else 0.01


    def calc_outputs(self, inputs):
        ## normalize inputs
        maximum = max(inputs)
        inputs = [input/maximum for input in inputs]
        self.inputs = inputs

        outputs = np.zeros(self.nodes_out)
        weighted_inputs = np.zeros(self.nodes_out)
        # net input = input_1 * weight_1 + ... + input_n * weight_n + bias
        for node_out in range(0, self.nodes_out):
            # add bias
            weighted_input = self.biases[node_out]

            for node_in in range(0, self.nodes_in):
                # multiply weight and input of each node and add it to the total (wich already includes bias)
                weighted_input += inputs[node_in] * self.weights[node_in, node_out]

            # use activation function to get output
            outputs[node_out] = self.activation_function(weighted_input)
            weighted_inputs[node_out] = weighted_input

        self.outputs = outputs
        self.weighted_inputs = weighted_inputs
        
        return outputs
    

    def node_cost(self, output_activation, expected_output):
        error = output_activation - expected_output
        return error * error
    

    def node_cost_derivative(self, output_activation, expected_output):
        # maybe just: output_activation - expected_output
        return 2 * (output_activation - expected_output)


    def calc_output_layer_node_vals(self, expected_outputs):
        # using the delta rule here
        node_values = np.zeros(len(expected_outputs))

        for i in range(0, len(node_values)):
            # cost derivative is how much the total error changes with respect to the output
            cost_derivative = self.node_cost_derivative(self.outputs[i], expected_outputs[i])
            # activation derivative is how much the output changes with respect to its total net input
            activation_derivative = self.activation_function_derivative(self.weighted_inputs[i])
            node_values[i] = activation_derivative * cost_derivative

        return node_values


    def update_gradients(self, node_values):
        for node_out in range(0, self.nodes_out):
            for node_in in range(0, self.nodes_in):
                derivative_cost_w_respect_to_weight = self.inputs[node_in] * node_values[node_out]
                self.cost_gradient_w[node_in, node_out] += derivative_cost_w_respect_to_weight

            derivative_cost_w_respect_to_bias = 1 * node_values[node_out]
            self.cost_gradient_b[node_out] += derivative_cost_w_respect_to_bias
        

    def calculate_hidden_layer_node_values(self, old_layer, old_node_values):
        new_node_values = np.zeros(self.nodes_out)
        # step 1: node cost deriv * avtivation deriv 
        # step 2 multply that by weight of connection between nodes
        # do 1 and 2 for each output neuron and add them together
        # step 3 get activation func deriv
        # step 4 multiply by input
        for new_node_idx in range(0, len(new_node_values)):
            new_node_val = 0

            for old_node_idx in range(0, len(old_node_values)):
                weighted_input_derivative = old_layer.weights[new_node_idx, old_node_idx]
                new_node_val += weighted_input_derivative * old_node_values[old_node_idx]

            new_node_val *= self.activation_function_derivative(self.weighted_inputs[new_node_idx])
            new_node_values[new_node_idx] = new_node_val

        return new_node_values


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)]

        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]


    def calc_outputs(self, inputs):
        for layer in self.layers:
            # repeat process for following layers using prev output as the next input
            inputs = layer.calc_outputs(inputs)

        # final result is output
        output = inputs
        return output
    

    def forward_pass(self, inputs):
        # the forward pass
        outputs = self.calc_outputs(inputs) 
        # After the forward pass we must calculate the loss/error (decision's quality)
        return outputs
    

    def individual_cost(self, data_point, output):
        # outputs = self.calc_outputs(data_point.inputs)
        output_layer = self.layers[-1]
        cost = 0

        for node_out in range(0, len(output)):
            cost += output_layer.node_cost(output[node_out], data_point.expected_outputs[node_out])

        return cost
    

    def cost(self, data, output):
        total_cost = 0
        for point in data:
            total_cost += self.individual_cost(point, output)

        return total_cost / len(data)
        # WANT LOWEST COST


    def learn(self, training_data, learn_rate): #
        for data_point in training_data:
            # backpropagation to calculate gradient of cost function for each point and and then the gradients are addded together
            self.update_all_gradients(data_point)

        self.apply_all_gradients(learn_rate)

        self.clear_all_gradients()

    
    def apply_all_gradients(self, learn_rate):
        for layer in self.layers:
            for node_in in range(0, layer.nodes_in):
                for node_out in range(0, layer.nodes_out):
                    layer.weights[node_in, node_out] -= learn_rate * layer.cost_gradient_w[node_in, node_out]
        
        for bias_idx in range(0, len(layer.biases)):
            layer.biases[bias_idx] -= learn_rate * layer.cost_gradient_b[bias_idx]


    def clear_all_gradients(self): #
        for layer in reversed(self.layers):
            layer.cost_gradient_w.fill(0)
            layer.cost_gradient_b.fill(0)


    def update_all_gradients(self, data_point): #
        # self.calc_outputs(data_point.inputs) # Dont think this needs to be here
        output_layer = self.layers[-1]
        node_values = output_layer.calc_output_layer_node_vals(data_point.expected_outputs)
        output_layer.update_gradients(node_values)

        for hidden_layer_idx in range(len(self.layers) - 2, -1, -1):
            hidden_layer = self.layers[hidden_layer_idx]
            node_values = hidden_layer.calculate_hidden_layer_node_values(self.layers[hidden_layer_idx + 1], node_values)
            hidden_layer.update_gradients(node_values)
