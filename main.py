from NeuralNetwork import NeuralNetwork
from GetInput import GetInput

# HAVE AN SVM CLASS THAT ACTS LIKE THE MODEL CLASS IN THE KERAS LIBRARY AND HAVE NODES THAT ARE SMALLER THAN A WHOLE LAYER AND MAKE THEM MORE MODULAR
# MAKE A CLASS FOR EVERY NODE - > IMPLEMENT THE GENERAL STEPS IN THE SVM CLASS AND HANDLE STUFF LIKE CALCULATING AND SAVING DERIVATIVES IN THE INSIDE OF THE NODES

# SVM - - > > MultiplyNode( Weight , Input ) - > HingleLossNode( Previous Output ) - > PlusNode( Previous Output , Regularization( Weight ) ) - - > > At The Beginning We Can Do It Without Regularization And Then Add It Later
# SVM - - > > L=1N∑i∑j≠yi[max(0,f(xi;W)j−f(xi;W)yi+1)]+αR(W) AND Li=∑j≠yi[max(0,wTjxi−wTyixi+1)]

# HYPERPARAMETERS WE CAN TWEAK IN HALF CROSS VALIDATION ARE STEP SIZE AND REG STRENGTH
# WE CAN PROBABLY NOT TWEAK EPOCHS BECAUSE THEY ARE GOING TO BE DEPENDENT ON STEP SIZE AND REGULARIZATION STRENGTH

# NUMPY NAN VALUES CAN BREAK THIS CODE SO IF THIS DOES NOT WORK NOW WE CAN TRY TO TWEAK BOTH OF THE LISTS SO THAT NONE OF THEM GO TO NUMPY NAN
# WE MIGHT GET AWAY WITH IT BECAUSE WE ARE COMPARING THE ACCURACY NUMBERS NOT THE LOSSES AND THE ACCURACY NUMBERS NEVER GOES TO NUMPY NAN


get_input = GetInput()

step_size_list = [ 1 , 0.1 , 0.01 , 0.001 ]
reg_strength_list = [ 0 ] # [ 0.1 , 0.01 , 0.001 , 0 ]

softmax_list = []

for now_step_size in step_size_list :
    for now_reg_strength in reg_strength_list :

        now_softmax = NeuralNetwork(real_input=True, X_train=get_input.X_train, y_train=get_input.y_train, step_size = now_step_size, reg_strength = now_reg_strength , y_test= get_input.y_test )
        now_softmax.add_layer("multiply", 784, 256)
        now_softmax.add_layer("plus", 256, 256)
        now_softmax.add_layer("relu")
        now_softmax.add_layer("multiply", 256, 128)
        now_softmax.add_layer("plus", 128, 128)
        now_softmax.add_layer("relu")
        now_softmax.add_layer("multiply", 128, 26)
        now_softmax.add_layer("plus", 26, 26)
        now_softmax.add_layer("relu")
        now_softmax.add_layer("multiply", 26, 26)
        now_softmax.add_layer("plus", 26, 26)
        now_softmax.add_layer("softmax")

        now_softmax.train_network(now_softmax.X, 1000 )
        now_softmax.test_softmax( now_softmax.X_validation , now_softmax.y_validation , "validation" )

        softmax_list.append( now_softmax )

best_softmax = sorted( softmax_list , key = lambda item : item.accuracy , reverse = True )[ 0 ]
best_softmax.test_softmax( get_input.X_test , get_input.y_test , "test")




"""
# RELATED INPUT SHAPE = 2 , RELATED OUTPUT SHAPE = 4

now_softmax.add_layer( "multiply" , 2 , 4 )
now_softmax.add_layer( "plus" , 4 , 4 )
now_softmax.add_layer( "relu" )
now_softmax.add_layer( "multiply" , 4 , 8 )
now_softmax.add_layer( "plus" , 8 , 8 )
now_softmax.add_layer( "relu" )
now_softmax.add_layer( "multiply" , 8 , 8 )
now_softmax.add_layer( "plus" , 8 , 8 )
now_softmax.add_layer( "relu" ) 
now_softmax.add_layer( "multiply" , 8 , 4 )
now_softmax.add_layer( "plus" , 4 , 4 ) 
now_softmax.add_layer( "softmax" )
now_softmax.train_network( now_softmax.X , 10000 )
now_softmax.test_softmax( now_softmax.X )
"""

"""
get_input = GetInput()
# now_softmax = Softmax()
now_softmax = Softmax( real_input= True , X_train = get_input.X_train , y_train = get_input.y_train , step_size = 0.1 , reg_strength = 0 )
# now_softmax.train_softmax( epoch = 80000 )

# RELATED INPUT SHAPE = 784 , RELATED OUTPUT SHAPE = 26 

now_softmax.add_layer( "multiply" , 784 , 256 )
now_softmax.add_layer( "plus" , 256 , 256 )
now_softmax.add_layer( "relu" ) 
now_softmax.add_layer( "multiply" , 256 , 128 )
now_softmax.add_layer( "plus" , 128 , 128 ) 
now_softmax.add_layer( "relu" ) 
now_softmax.add_layer( "multiply" , 128 , 26 ) 
now_softmax.add_layer( "plus" , 26 , 26 ) 
now_softmax.add_layer( "relu" ) 
now_softmax.add_layer( "multiply" , 26 , 26 ) 
now_softmax.add_layer( "plus" , 26 , 26 )
now_softmax.add_layer( "softmax" ) 

now_softmax.train_network( now_softmax.X , 1000 )
now_softmax.test_softmax( get_input.X_test , get_input.y_test ) 
"""

