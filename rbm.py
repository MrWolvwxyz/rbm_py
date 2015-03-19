import os, struct, sys
from array import array as pyarray
from time import sleep
from pylab import *
from numpy import *

def logistic( x ):
    return 1.0 / ( 1.0 + exp( -x ) )
    
def init_activation( x ):
    #For teh dropout
    #mask = 0.75 > random.rand( x.shape[ 0 ], x.shape[ 1 ] )
    logis = logistic( x )
    return logis, logis > random.rand( x.shape[ 0 ], x.shape[ 1 ] )

def drop_activation( x ):
    #For teh dropout
    mask = 0.5 > random.rand( x.shape[ 0 ], x.shape[ 1 ] )
    logis = logistic( multiply( x, mask ) )
    return logis, logis > random.rand( x.shape[ 0 ], x.shape[ 1 ] )

def load_mnist( dataset = "training", digits = arange( 10 ),
                path = "/Users/Sam/Documents/rbm_py/" ):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows * cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows * cols))
        labels[i] = lbl[ind[i]]

    return images, labels

num_visible = 784
num_hidden = 1000
batch = 100
set_size = 60000
rate = 0.05
sp_par = 0.001
max_epoch = 10
init_mom = 0.5
final_mom = 0.9
weight_cost = 0.001

images, labels = load_mnist( 'training' )
tr_images = images / 255.0

weights = 0.001 * random.randn( num_visible, num_hidden )
hid_bias = zeros( num_hidden )
vis_bias = zeros( num_visible )

w_grad = zeros( ( num_visible, num_hidden ) )
hb_grad = zeros( num_hidden )
vb_grad = zeros( num_visible )

for i in range( num_visible ):
    if vis_bias[ i ] == -inf:
        vis_bias[ i ] = 0
 
print( "training standard rbm" )       
for i in range( max_epoch ):
   for tr_batch in range( set_size / batch ):
        print( "epoch ", i, " batch ", tr_batch )
        num_case = set_size / batch 
        momentum = 0
        
        #positive phase
        data = tr_images[ batch * tr_batch : batch * ( 1 + tr_batch ), : ] > rand( batch, num_visible )
        hid_prob, hid_act = init_activation( hid_bias + 2 * dot( data, weights ) )
        
        #negative phase
        vis_prob, vis_act = init_activation( vis_bias + dot( hid_act, transpose( weights ) ) )
        hid_rec_prob, hid_rec_act = init_activation( hid_bias + 2 * dot( vis_act, weights ) )
            
        #update momentum
        if i > 5:
            momentum = final_mom
        else:
            momentum = init_mom
        
        #compute gradients and add them to the parameters
        w_grad = w_grad * momentum + rate * ( ( dot( transpose( data ), hid_prob )
                          - dot( transpose( vis_act ), hid_rec_prob ) ) / num_case - weight_cost * weights )
        vb_grad = momentum * vb_grad + rate * ( data - vis_act ).mean( axis = 0 )
        hb_grad = momentum * hb_grad + rate * ( hid_prob - hid_rec_prob ).mean( axis = 0 )
        
        #update parameters
        weights += w_grad
        vis_bias += vb_grad
        hid_bias += hb_grad
        
        #display
        if( ( tr_batch == ( set_size / batch - 1 ) and i > 5 )
         or ( tr_batch == 0 and i == 0 ) ):
            fig = figure()
            a = fig.add_subplot( 3, 2, 1 )
            imshow( vis_act[ 3 ].reshape( 28, 28 ), cmap = cm.gray )
            a = fig.add_subplot( 3, 2, 2 )
            imshow( vis_act[ 0 ].reshape( 28, 28 ), cmap = cm.gray )
            a = fig.add_subplot( 3, 2, 3 )
            imshow( vis_act[ 4 ].reshape( 28, 28 ), cmap = cm.gray )
            a = fig.add_subplot( 3, 2, 4 )
            imshow( vis_act[ 1 ].reshape( 28, 28 ), cmap = cm.gray )
            a = fig.add_subplot( 3, 2, 5 )
            imshow( vis_act[ 5 ].reshape( 28, 28 ), cmap = cm.gray )
            a = fig.add_subplot( 3, 2, 6 )
            imshow( vis_act[ 2 ].reshape( 28, 28 ), cmap = cm.gray )
            show()

print( "training crbm" )            
            
'''
print( "training next layer" )
#do it again for the next layer!
num_hid2 = 784
weights2 = 0.001 * random.randn( num_hidden, num_hid2 )
hid2_bias = zeros( num_hid_2 )
cd = ceil( max_epoch / 20 )

w1_grad = zeros( ( num_hidden, num_hid2 ) )
w2_grad = zeros( ( num_visible, num_hidden ) )
hb_grad2 = zeros( num_hid2 )
vb_grad2 = zeros( num_hidden )


for i in range( max_epoch ):
   for tr_batch in range( set_size / batch ):
        num_case = set_size / batch 
        momentum = 0
        
        #need values for hidden layer
        data = tr_images[ batch * tr_batch : batch * ( 1 + tr_batch ), : ] > rand( batch, num_visible )
        hid_prob, hid_act = init_activation( hid_bias + 2 * dot( data, weights ) )
        
        #positive phase
        hid2_prob, hid2_act = init_activation( hid_bias2 + 2 * dot( hid_act, weights2 ) )
        
        hid_prob_temp = hid2_prob
        
        #initialize these here for use in updates
        hid_rec_prob, hid_rec_act = zeros( ( batch, num_hidden ) )
        hid2_rec_prob, hid2_rec_act = zeros( ( batch, num_hid2 ) )
        #do gibbs sampling cd times at top layer
        for i in range( cd ):
            #negative phase
            hid_states = hid_prob_temp > rand( batch, num_hidden )
            
            hid_rec_prob, hid_rec_act = drop_activation( hid_bias + dot( hid_states, transpose( weights2 ) ) )
            hid2_rec_prob, hid2_rec_act = drop_activation( hid2_bias + 2 * dot( vis_act, weights2 ) )
            hid_prob_temp = hid2_rec_prob
            
        #update momentum
        if i > 5:
            momentum = final_mom
        else:
            momentum = init_mom
        
        #compute gradients and add them to the parameters
        w1_grad = w1_grad * momentum + rate * ( ( dot( transpose( hid_act ), hid2_prob )
                          - dot( transpose( hid_rec_act ), hid2_rec_prob ) ) / num_case - weight_cost * weights )
        w2_grad = w2_grad * momentum + rate * ( ( dot( transpose( data ), hid2_rec_prob )
                          - dot( transpose( vis_act ), hid_rec_prob ) ) / num_case - weight_cost * weights2 )
        vb_grad = momentum * vb_grad + rate * ( data - vis_act ).mean( axis = 0 )
        hb_grad = momentum * hb_grad + rate * ( hid_prob - hid_rec_prob ).mean( axis = 0 )
        
        #update parameters
        weights += w1_grad
        vis_bias += vb_grad
        hid_bias += hb_grad

        #display
        if tr_batch == ( set_size / batch - 1 ) or ( tr_batch == 0 and i == 0 ):
            fig = figure()
            a = fig.add_subplot( 3, 2, 1 )
            imshow( vis_act[ 3 ].reshape( 28, 28 ), cmap = cm.gray )
            a = fig.add_subplot( 3, 2, 2 )
            imshow( vis_act[ 0 ].reshape( 28, 28 ), cmap = cm.gray )
            a = fig.add_subplot( 3, 2, 3 )
            imshow( vis_act[ 4 ].reshape( 28, 28 ), cmap = cm.gray )
            a = fig.add_subplot( 3, 2, 4 )
            imshow( vis_act[ 1 ].reshape( 28, 28 ), cmap = cm.gray )
            a = fig.add_subplot( 3, 2, 5 )
            imshow( vis_act[ 5 ].reshape( 28, 28 ), cmap = cm.gray )
            a = fig.add_subplot( 3, 2, 6 )
            imshow( vis_act[ 2 ].reshape( 28, 28 ), cmap = cm.gray )
            show()
'''




