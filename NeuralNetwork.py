import numpy as np
import matplotlib.pyplot as plt
import sys

class MultiplyNode :
    def __init__( self , input_size = 200  , layer_size = 4 , dimensionality = 2 , step_size = 1.0 , reg_strength = 0.001 ) :

        # DESIGNED FOR INPUT AND WEIGHT MATRIX MULTIPLICATION AND UPDATES THE MATRIX WEIGHTS AUTOMATICALLY

        self.input_size = input_size
        self.layer_size = layer_size
        self.dimensionality = dimensionality
        self.step_size = step_size


        self.reg_strength = reg_strength
        self.weight_matrix = 0.01 * np.random.randn(self.dimensionality, self.layer_size) # DOUBLE NOT RELATED

    def forward_pass( self , X , pass_type = "train" ) :

        # INPUT = X [ INPUT MATRIX ] - > OUTPUT = scores [ OUTPUT MATRIX ]
        # COMPUTE THE CLASS SCORES FOR EACH INPUT

        scores = np.dot( X, self.weight_matrix )
        self.X = X
        return scores

    def backward_pass( self , dZ  , pass_type = "train" ) :

        # INPUT = dZ [ GRADIENT OF THE STEP FORWARD IN THE COMPUTATIONAL GRAPH ] - > OUTPUT = GRADIENT ON INPUTS
        # REGULARIZATION GETS APPLIED HERE TO THE GRADIENT OF THE WEIGHT MATRIX

        # CALCULATE THE GRADIENT OF THE WEIGHT MATRIX
        dW = np.dot( self.X.T , dZ )

        # THIS IS L2 REGULARIZATION
        dW += self.reg_strength * np.linalg.norm( self.weight_matrix ) # BE CAREFUL

        # CALCULATE THE GRADIENT OF THE INPUT MATRIX WHICH IS NEEDED FOR BACK PROPAGATION AS IT CAN BE COMING FROM THE PREVIOUS LAYER
        dX = np.dot( dZ , self.weight_matrix.T )

        # CHANGE THE WEIGHTS AND GO TO THE OPPOSITE DIRECTION OF THE STEEPEST GRADIENT
        self.weight_matrix += self.step_size * dW * ( - 1 )

        # RETURN THE GRADIENT ON THE INPUT MATRIX FOR BACK PROPAGATION
        return dX

class SoftmaxLossNode :
    def __init__( self , y , input_size , reg_strength = 0 , weight_matrix_list = [] , y_validation = [] , y_test = [] ) :


        self.y = y
        self.input_size = input_size
        self.reg_strength = reg_strength
        self.weight_matrix_list = weight_matrix_list
        self.y_validation = y_validation
        self.y_test = y_test

    def forward_pass( self , scores , pass_type = "train" ) :

        # PRINTS THE LOSS
        # INPUT = scores [ CLASS SCORES VECTOR FOR EACH INPUT SO IT IS A MATRIX ] - > OUTPUT - > probs [ CLASS PROBABILITY VECTOR FOR EACH INPUT SO IT IS A MATRIX ]

        # GET CLASS PROBABILITIES
        # SCORES DIMENSION = [ 200 , 4 ] , PROBABILITY DIMENSION = [ 200 , 4 ]

        exp_scores = np.exp( scores )
        probs = exp_scores / np.sum( exp_scores , axis = 1 , keepdims = True )

        # COMPUTE THE LOSS HERE AND MAKE IT SOFT MAX LOSS BECAUSE IT IS EASIER TO DIFFERENTIATE
        # BE CAREFUL # BE CAREFUL # BE CAREFUL

        """
        correct_logprobs = []
        for i in range( len( probs ) ) :
            # print( f" i = { i } , self.y[ i ] - 1 = { self.y[ i ] - 1 } " )
            correct_prob = probs[ i ][ self.y[ i ] - 1 ]
            incorrect_prob_sum = 0
            for j in range( len( probs[ i ] ) ) :
                if j != self.y[ i ] :
                    incorrect_prob_sum += probs[ i ][ j ]
            final_prob = correct_prob / incorrect_prob_sum
            log_prob = - np.log( final_prob )
            correct_logprobs.append( log_prob )
        """

        # SELF Y IS ACTUALLY ONLY THE TRAINING AFTER THE SPLIT HAS MADE
        if pass_type == "train" :
            correct_logprobs = - np.log( probs[ range( self.input_size )  , self.y] ) # BE CAREFUL
            data_loss = np.sum(correct_logprobs) / self.input_size
        elif pass_type == "validation" :
            correct_logprobs = - np.log(probs[range( len( probs ) ), self.y_validation ])  # BE CAREFUL
            data_loss = np.sum(correct_logprobs) / len( probs )
        elif pass_type == "test" :
            correct_logprobs = - np.log( probs[ range( len( probs ) ) , self.y_test ] ) # BE CAREFUL
            data_loss = np.sum(correct_logprobs) / len(probs)

            # COMPUTES THE L2 REGULARIZATION AND ADDS THAT TO THE SUM LOSS
        sum_weight_matrix_norm = 0
        for weight_matrix in self.weight_matrix_list :
            # print( f" weight_matrix = { weight_matrix } " )
            now_sum = np.sum( np.linalg.norm( weight_matrix ) )
            sum_weight_matrix_norm += now_sum


        reg_loss = 0.5 * self.reg_strength * sum_weight_matrix_norm # LEARN THIS
        sum_loss = data_loss + reg_loss

        print( f" Sum Loss = { sum_loss } , Reg Loss = { reg_loss } , Data Loss = { data_loss } " )
        return probs

    def backward_pass( self , probs , pass_type = "train" ) :

        # INPUT = probs [ CLASS PROBABILITIES MATRIX ] - > OUTPUT = dscores [ GRADIENT OF CLASS SCORES ON THE CLASS PROBABILITIES ] # BE CAREFUL
        # DIFFERENTIATE THE SOFT MAX LOSS FUNCTION

        # ALL THE CLASS PROBABILITIES STAY THE SAME EXCEPT SUBTRACT ONE FOR THE CORRECT CLASS GRADIENT
        # MAKE SENSE BECAUSE IF YOU INCREASE THE PROBABILITY OF A WRONG CLASS THEN THE LOSS INCREASES BUT IF YOU INCREASE THE PROBABILITY OF THE CORRECT CLASS THE LOSS DECREASES

        dscores = np.array( probs ) # CHANGED

        # BE CAREFUL # BE CAREFUL # BE CAREFUL

        """
        for i in range( len( dscores ) ) :
            correct_class_index = self.y[ i ] - 1
            # print( f" i = { i } , self.y[ i ] - 1  = { self.y[ i ] - 1 } , dscores[ i ][ correct_class_index ] = { dscores[ i ][ correct_class_index ] } " )
            dscores[ i ][ correct_class_index ] -= 1 
        """

        dscores[ range(self.input_size) , self.y ] -= 1 # BE CAREFUL
        dscores /= self.input_size

        return dscores

class PlusNode :
    def __init__(self, dimensionality =4, step_size=1.0 ):

        self.dimensionality = dimensionality
        self.step_size = step_size
        self.bias_vector = np.zeros( ( 1 , self.dimensionality ) )

    def forward_pass( self , scores , pass_type = "train" ) :

        # INPUT = scores [ SCORES MATRIX AFTER THE WEIGHT AND INPUT MULTIPLICATION ] - > OUTPUT = scores [ AFTER BIAS ADDED FOR EACH SCORE ] # PROBABLY WORKS

        scores += self.bias_vector
        return scores


    def backward_pass( self , dscores , pass_type = "train" ) :

        # INPUT = dscores [ FORWARD NODE GRADIENT MATRIX ] - > OUTPUT = dscores [ SAME MATRIX BUT WE CHANGE THE BIAS VECTOR INSIDE THIS METHOD ]

        db = np.sum( dscores , axis=0 , keepdims = True )  # AXIS 0 = COL
        self.bias_vector += self.step_size * db * ( - 1 )
        return dscores

class ReluNode :
    def __init__( self ) :
        pass

    def forward_pass( self , scores , pass_type = "train" ) :
        self.scores = scores
        return np.maximum( 0 , scores )

    def backward_pass( self , dZ , pass_type = "train" ) :
        dZ[ self.scores <= 0 ] = 0
        return dZ

class NeuralNetwork :
    def __init__( self , N_IN = 100 , D_IN = 2 , K_IN = 4 , step_size = 0.1  , reg_strength = 0 , real_input = False , X_train = None , y_train = None , y_test = None ) :

        if real_input == False :
            return
        else :

            # HANDLE THE REAL INPUT DETAILS HERE IF THE REAL INPUT IS PROVIDED

            # TRAIN VALIDATION SPLITS

            self.X = X_train[ : 66600 ]
            self.y = y_train[ : 66600 ]
            self.X_validation = X_train[ 66600 : ]
            self.y_validation = y_train[ 66600 : ]

            self.number_of_classes = 26
            self.dimensionality = 784 # len( self.X[ 0 ] )
            self.input_size = len( self.X ) # len( self.X )

            self.step_size = step_size
            self.reg_strength = reg_strength

            self.layer_list = []
            self.weight_matrix_list = []
            self.y_test = y_test

    """
    def train_softmax( self , epoch ) :

        # THERE IS A BUG WHEN WE GO TO HIGH EPOCHS IT OUTPUTS NONE LOSSES AND PROBABLY BECAUSE IT GETS INFINITE SOMEWHERE AND PROBABLY COULD BE BECAUSE NUMBERS EXPLODE OR SOMETHING LIKE THAT IT WAS SAID IN CS231N

        # SO BASICALLY THE PREVIOUS NODES LAYER SIZE EQUALS NOW NODES DIMENSIONALITY AND EXCEPT THE LAST COMBINED LAYER THE LAYER SIZE DOES NOT MATTER FOR THE NETWORK TO AT LEAST WORK BUT THE LAST COMBINED LAYER MUST HAVE THE NUMBER OF CLASSES WE ARE TRYING TO CLASSIFY

        # I THINK THAT WHEN ADDING A NEW MULTIPLY NODE NODE INSIDE THE NETWORK OR IN THE BEGINNING WE DO NOT NEED IT TO BE SAME TO NUMBER OF CLASSES THE IMPORTANT PART IS THE END BEFORE THE SOFTMAX

        self.node_one = MultiplyNode( self.input_size , 4 , self.dimensionality , self.step_size , self.reg_strength ) # ( self , input_size = 200  , layer_size = 4 , dimensionality = 2 , step_size = 1 , reg_strength = 0.001 )

        # WE DO NOT NEED TO UPDATE THE LIST WITH UPDATED MATRIXES EVERY TIME BECAUSE THEY ARE PROBABLY REFERENCES AND IF ONE CHANGES THE OTHER CHANGES AS WELL
        # self.weight_matrix_list.append( self.node_one.weight_matrix )

        self.node_two = PlusNode( 4 , self.step_size ) # (self, number_of_classes=4, step_size=1.0 ) # BE CAREFUL
        self.relu_node = ReluNode()
        self.node_post_two = MultiplyNode( self.input_size , self.number_of_classes , 4 , self.step_size , self.reg_strength ) # ( self , input_size = 200  , layer_size = 4 , dimensionality = 2 , step_size = 1 , reg_strength = 0.001 )

        # WE DO NOT NEED TO UPDATE THE LIST WITH UPDATED MATRIXES EVERY TIME BECAUSE THEY ARE PROBABLY REFERENCES AND IF ONE CHANGES THE OTHER CHANGES AS WELL
        # self.weight_matrix_list.append(self.node_post_two.weight_matrix )

        self.node_post_post_two = PlusNode( self.number_of_classes , self.step_size ) # (self, number_of_classes=4, step_size=1.0 ) # BE CAREFUL
        self.node_three = SoftmaxLossNode( self.y , self.input_size , self.reg_strength , self.weight_matrix_list ) # ( self , y )

        for i in range( epoch ) :

            post_dot_product = self.node_one.forward_pass( self.X ) # ( self , X )
            post_added_bias = self.node_two.forward_pass( post_dot_product ) # ( self , scores )
            post_relu_node = self.relu_node.forward_pass( post_added_bias )
            trial_step_1 = self.node_post_two.forward_pass(post_relu_node)  # ( self , X )
            trial_step_2 = self.node_post_post_two.forward_pass( trial_step_1 ) # ( self , scores )
            post_softmax = self.node_three.forward_pass( trial_step_2 ) # ( self , scores )

            post_back_softmax_loss = self.node_three.backward_pass(post_softmax)
            trial_step_back_2 = self.node_post_post_two.backward_pass(post_back_softmax_loss)  # ( self , scores )
            trial_step_back_1 = self.node_post_two.backward_pass(trial_step_back_2)  # ( self , X )
            post_back_relu_node = self.relu_node.backward_pass( trial_step_back_1 )
            post_back_added_bias = self.node_two.backward_pass( post_back_relu_node )
            post_back_multiply_node = self.node_one.backward_pass(post_back_added_bias)

        print( self.node_one.weight_matrix )
        print( self.node_post_two.weight_matrix )
    """

    def add_layer( self , type , dimensionality = None , layer_size = None ) :

        # MULTIPLY NODE = ( self , input_size = 200  , layer_size = 4 , dimensionality = 2 , step_size = 1 , reg_strength = 0.001 )
        # PLUS NODE = (self, dimensionality =4, step_size=1.0 )
        # RELU NODE = ()
        # SOFTMAX NODE = ( self , y , input_size , reg_strength = 0 , weight_matrix_list = [] )

        # PREVIOUS COMBINED NODE LAYER SIZE = NOW COMBINED NODE DIMENSIONALITY
        # FIRST COMBINED NODE DIMENSIONALITY = INPUT DIMENSIONALITY
        # LAST COMBINED NODE LAYER SIZE = NUMBER OF CLASSES REQUIRED
        # INPUT SIZE AND STEP SIZE AND REGULARIZATION STRENGTH STAYS THE SAME FOR ALL SO ONLY INPUTS WE NEED MOST OF THE TIMES ARE LAYER SIZE AND DIMENSIONALITY
        # DO IT LIKE THIS - > DIMENSIONALITY , LAYER SIZE - > BECAUSE IT IS RELATED TO HOW THE DIMENSIONS CHANGES

        if type == "multiply" :
            now_node = MultiplyNode( self.input_size , layer_size , dimensionality , self.step_size , self.reg_strength )
            self.weight_matrix_list.append( now_node.weight_matrix )
            self.layer_list.append( now_node )

        elif type == "plus" :
            now_node = PlusNode( dimensionality , self.step_size )
            self.layer_list.append( now_node )

        elif type == "relu" :
            now_node= ReluNode()
            self.layer_list.append( now_node )

        elif type == "softmax" :
            now_node = SoftmaxLossNode( self.y , self.input_size , self.reg_strength , self.weight_matrix_list , self.y_validation , self.y_test )
            self.layer_list.append( now_node )

    def train_network( self , input_network , epoch ) :
        self.accuracy_list = []
        for i in range( epoch ) :

            dZ = self.forward_pass_network( input_network )
            predicted_classes = np.argmax( dZ , axis = 1 )
            self.accuracy_list.append( np.mean( predicted_classes == self.y ) )
            print( f" Iteration = { i } , Training Accuracy = { np.mean( predicted_classes == self.y ) } " )
            dX = self.backward_pass_network( dZ )

            if i > 10 :
                if self.accuracy_list[ i ] == self.accuracy_list[ i - 10 ] :
                    break

    def forward_pass_network( self , input_network , type = "train" ) :
        prev_input = np.array( input_network )
        for i in range( len( self.layer_list ) ) :
            prev_input = self.layer_list[ i ].forward_pass( prev_input , pass_type = type  )
        return prev_input

    def backward_pass_network( self , output_probs ) :
        dZ = np.array( output_probs )
        for i in range( len( self.layer_list ) - 1 , - 1 , - 1 ) :
            dZ = self.layer_list[ i ].backward_pass( dZ )
        return dZ

    def test_softmax( self , input_data , ground_truth , test_type ) :
        probs = self.forward_pass_network( input_data , type = test_type )
        predicted_classes = np.argmax( probs , axis = 1 )

        if test_type == "validation" :
            print(" ________________________________________________________________________________ ")
            print(f" Iteration = Validation , Training Accuracy = {np.mean(predicted_classes == ground_truth)} ")
            print(" ________________________________________________________________________________ ")

        elif test_type == "test" :
            print( " ________________________________________________________________________________ " )
            print(f" Iteration = Test , Training Accuracy = {np.mean(predicted_classes == ground_truth)} ")
            print(" ________________________________________________________________________________ ")

        self.accuracy = np.mean( predicted_classes == ground_truth )

