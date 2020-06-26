import numpy as np

# import pandas as pd
# import matplotlib.pyplot as plt

# ## Import bikesharing data
# data_path = 'Bike-Sharing-Dataset/hour.csv'
# rides = pd.read_csv(data_path)
# rides.head()
# # temp = rides.head(100)

# ## Quick look at cnt for first 10 days
# rides[:24*10].plot(x='dteday', y='cnt')

# ## Create dummy variables for categorical variable (One-Hot encoding)
# dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
# for each in dummy_fields:
#     dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
#     rides = pd.concat([rides, dummies], axis=1)

# fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
#                   'weekday', 'atemp', 'mnth', 'workingday', 'hr']
# data = rides.drop(fields_to_drop, axis=1)
# data.head()

# ## Standardize continuous variables (0 mean and 1 sd)
# quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# # Store scalings in a dictionary so we can convert back later
# scaled_features = {}
# for each in quant_features:
#     mean, std = data[each].mean(), data[each].std()
#     scaled_features[each] = [mean, std]
#     data.loc[:, each] = (data[each] - mean)/std
    
# ## Split data between training, testing and validation
# # Save data for approximately the last 21 days 
# test_data = data[-21*24:]

# # Now remove the test data from the data set 
# data = data[:-21*24]

# # Separate the data into features and targets
# target_fields = ['cnt', 'casual', 'registered']
# features, targets = data.drop(target_fields, axis=1), data[target_fields]
# test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# # Hold out the last 60 days or so of the remaining data as a validation set
# train_features, train_targets = features[:-60*24], targets[:-60*24]
# val_features, val_targets = features[-60*24:], targets[-60*24:]



class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.matmul(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = error * self.weights_hidden_to_output
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error
        
        hidden_error_term = hidden_error.T *  hidden_outputs * (1 - hidden_outputs)
        
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:,None]
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:,None]
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.matmul(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 2500
learning_rate = 0.9
hidden_nodes = 10
output_nodes = 1

## default
# iterations = 100
# learning_rate = 0.1
# hidden_nodes = 2
# output_nodes = 1