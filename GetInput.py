from numpy import genfromtxt
import numpy as np

class GetInput :
    def __init__( self ) :
        train_data = genfromtxt('archive/emnist-letters-train.csv', delimiter=',')
        test_data = genfromtxt('archive/emnist-letters-test.csv', delimiter=',')

        self.X_train, self.y_train = self.make_X_and_y(train_data)
        self.X_test, self.y_test = self.make_X_and_y(test_data)

        print( f" len( X_train ) = { len( self.X_train ) } , len( X_train[ 0 ] ) = { len( self.X_train[ 0 ] ) } " )
        print( f" len( X_test ) = { len( self.X_test ) } , len( X_test[ 0 ] ) = { len( self.X_test[ 0 ] ) } " )


        print( f" len( y_train ) = { len( self.y_train ) } , len( y_test ) = { len( self.y_test ) } " )
        # LOOKS GOOD

    def make_X_and_y( self , data_matrix ) :
        y = []
        X = []
        for row in data_matrix :
            y_now = int( row[ 0 ] - 1  )
            x_now = []
            for i in range( 1 , len( row ) ) :
                x_now.append( row[ i ] )
            y.append( y_now )
            X.append( x_now )
        return [ np.array( X ) , np.array( y ) ] # CHANGED

