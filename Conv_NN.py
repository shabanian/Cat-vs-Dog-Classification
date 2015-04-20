import numpy as np 
import  scipy
import  scipy.signal
import scipy.ndimage
import cPickle
import theano.tensor.nnet.conv as conv
import theano.tensor.signal.downsample
import theano.tensor.shared_randomstreams
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy
import pylab
from logistic_sgd import LogisticRegression
from sklearn import svm
from sklearn.svm import NuSVC
import theano
import theano.tensor as T
from load_data import load_color
 
 
def gradient_updates_momentum(cost, params, learning_rate, momentum):
    assert momentum < 1 and momentum >= 0
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    return updates
    
def relu(x):
    return x * (x > 1E-6)
    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=relu):
 
        self.input = input
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-0.1,
                    high= 0.1, 
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
 
        self.params = [self.W, self.b]
class  LeNetConvPoolLayer( ):
	def __init__(self,rng,  input,  image_shape, filter_shape, poolsize=(2, 2),
                     activation =T.tanh, stride = (1,1)): 
		self.rng =rng
        	assert image_shape[1] == filter_shape[1]
        	W_bound = 0.05 
		W = theano.shared( np.asarray( self.rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX ), borrow=True)
		 
        	self.W =  W 
      
        	
        	S = ( filter_shape[0], image_shape[2]- filter_shape[2] + 1, image_shape[2]- filter_shape[2] + 1) 
 
        	b_values = np.zeros(S, dtype=theano.config.floatX)
        	self.b = theano.shared(value=b_values, borrow=True)
         
        	conv_out = activation(theano.sandbox.cuda.dnn.dnn_conv(
            	img=input,
            	kerns=  self.W  ) + self.b) 	
         
         
		self.output = theano.sandbox.cuda.dnn.dnn_pool(img= conv_out, ws=poolsize, stride=stride, mode='max', pad=(0, 0))
        	 
        	self.params = [self.W, self.b] 
 
 

def CCD(Number_Conv_Layer = 4 ,
        Number_Conv_feature = [32,32,64,128] ,
        Kernel_size = [3,3,5,5],
        Activation_Conv = [relu,relu,relu, relu],#[T.tanh,T.tanh,T.tanh, T.tanh],# 
        pooling_size = [3,3,3,3],
        stride = [2,2,2,2],
        Number_Hidden_Layer = 3 ,
        Number_Hidden_feature = [256, 324, 128] ,
        Activation_Hidden = [relu,relu,relu ],#[T.tanh,T.tanh,T.tanh ],#
        learning_rate =  0.01,
        momentum = 0.9,
        batch_size = 50,
        input_size = 256,
        Number_Classes = 2, 
        n_epochs  = 300,
        N_train_example = 22500,
        N_test_example = 2500 ,
        color = True,
        ):
		  
    print '... building the model'
    if color:
        c = 3
    else:
        c =1
    rng = np.random.RandomState( )
    x = T.ftensor4('x')
    layer_input_size = (batch_size, c, input_size, input_size) 
    Next_layer_input = x.reshape(layer_input_size)
    Filter_shape = (Number_Conv_feature[0], c,Kernel_size[0], Kernel_size[0] )
    
       
    Conv_layer =[]
    Params =[]
    for i in range(Number_Conv_Layer):
    
        Conv_layer.append([])
        Conv_layer[i]= LeNetConvPoolLayer(
			    rng,
			    input = Next_layer_input ,
			    image_shape= layer_input_size,
			    filter_shape= Filter_shape,
			    poolsize=(pooling_size[i], pooling_size[i]),
			    activation =Activation_Conv[i] ,
			    stride =( stride[i], stride[i]))
	if i ==0:
	  zeros = T.zeros([batch_size,Number_Conv_feature[0] , 127,127 ], dtype='float32')
	  Conv_layer[0].output = T.set_subtensor(zeros[:,:,:126,:126], Conv_layer[0].output)
	if i ==2:
	  zeros = T.zeros([batch_size,Number_Conv_feature[i] ,   29, 29 ], dtype='float32')
	  Conv_layer[i].output = T.set_subtensor(zeros[:,:,:28,:28], Conv_layer[i].output)
        if Number_Conv_Layer-1 != i:
            Filter_shape = (Number_Conv_feature[i+1], Filter_shape[0],Kernel_size[i+1], Kernel_size[i+1] )
             
        layer_input_size =  (batch_size, Number_Conv_feature[i],1+int( np.floor(((layer_input_size[2] - Kernel_size[i] + 1)-pooling_size[i]+1)/(1.0*stride[i]))),
                            1+int(np.floor(((layer_input_size[2] - Kernel_size[i] + 1)-pooling_size[i]+1)/(1.0*stride[i]))))
        Next_layer_input  = Conv_layer[i].output 
        Params += Conv_layer[i].params
        

   
    
    Next_layer_input = Next_layer_input.flatten(2)
    layer_input_size = (batch_size, layer_input_size[1]*layer_input_size[2]*layer_input_size[3])
    
    Hidden_layer =[]
    for i in range(Number_Hidden_Layer):
        Hidden_layer.append([])
        Hidden_layer[i]= HiddenLayer(
            rng=rng,
            input=Next_layer_input,
            n_in= layer_input_size[1],
            n_out= Number_Hidden_feature[i],
            activation=Activation_Hidden[i]
                ) 
    
        layer_input_size = (batch_size,Number_Hidden_feature[i] )
  
        Next_layer_input  = Hidden_layer[i].output 
        Params += Hidden_layer[i].params
  
    X_t, y_t = load_color()			
    
 
    y = T.ivector('y') 
    logRegressionLayer = LogisticRegression(
            input=Next_layer_input,
            n_in=layer_input_size[1],
            n_out=Number_Classes 
        )
    Params += logRegressionLayer.params
    Reg = 0
    Reg_cov = 0
    ind_param =[8,10,12]
    for pp in ind_param:
 		Reg += T.sum(abs(Params[pp])) 
    
    cost = logRegressionLayer.negative_log_likelihood( y) + 0.001*Reg  
    #gparams = T.grad(cost,  Params )
    learning_r = T.fscalar()
    updates =gradient_updates_momentum(cost, Params , learning_r   , momentum )
 
    Train_function = theano.function(
        inputs=[x,y, learning_r ],
        outputs=cost,
        updates=updates
        )
      
    Get_Error = theano.function(
                [x,y],
                logRegressionLayer.errors(y) 
                )
    Get_NLL = theano.function(
                [x,y],
                cost
                )
 
 
    print '... training'
 
    n_train_batches =  N_train_example /batch_size
    n_test_batches =  N_test_example /batch_size
    Test_list_scores = []
    Train_list_scores = []
    layer_input_size = (batch_size, c, input_size, input_size) 
    for epoch in range( n_epochs):
        
        print '--- epoch: ', epoch
                 
        minibatch_avg_cost_total  = 0.0
        for minibatch_index in xrange(n_train_batches):
                srng = np.random.RandomState(rng.randint(999999))
             
                train_set_x=  np.array(X_t[minibatch_index * batch_size:(minibatch_index + 1) * batch_size], dtype='float32') /np.array(X_t[minibatch_index * batch_size:(minibatch_index + 1) * batch_size], dtype='float32').max()
          
                train_set_y=  np.array(y_t[minibatch_index * batch_size:(minibatch_index + 1) * batch_size] , dtype='int32') 
         
              
 
                minibatch_avg_cost = Train_function(train_set_x,train_set_y, learning_rate)
                minibatch_avg_cost_total += minibatch_avg_cost
        
 
        test_losses = [Get_Error(
                              np.array(X_t[N_train_example  + i * batch_size:N_train_example + (i + 1) * batch_size].reshape(layer_input_size), dtype='float32')/np.array(X_t[N_train_example  + i * batch_size:N_train_example + (i + 1) * batch_size], dtype='float32').max(), 
                              np.array(y_t[N_train_example+ i * batch_size:N_train_example + (i + 1) * batch_size] , dtype='int32') 
                                     ) 
                              for i in xrange(n_test_batches)
                        ]
        test_NLL = [Get_NLL(
                              np.array(X_t[N_train_example  + i * batch_size:N_train_example + (i + 1) * batch_size].reshape(layer_input_size), dtype='float32')/np.array(X_t[N_train_example  + i * batch_size:N_train_example + (i + 1) * batch_size] , dtype='float32').max(), 
                              np.array(y_t[N_train_example+ i * batch_size:N_train_example + (i + 1) * batch_size] , dtype='int32') 
                                     ) 
                              for i in xrange(n_test_batches)
                        ]
	
        this_test_loss = np.mean(test_losses)
 
	
	#learning_rate *= 0.995
        """if learning_rate> 0.0005:
	            learning_rate *= 0.95
        else:
		    learning_rate = 0.0005"""
        Test_list_scores.append(float(this_test_loss))
        print '........................Test error:' ,this_test_loss
                
        train_loss = [Get_Error(
                              np.array(X_t[ i * batch_size: (i + 1) * batch_size].reshape(layer_input_size), dtype='float32')/np.array(X_t[ i * batch_size: (i + 1) * batch_size], dtype='float32').max(),  
                              np.array(y_t[   i * batch_size: (i + 1) * batch_size] , dtype='int32') 
                                     ) 
                              for i in xrange(n_train_batches)
                        ]
        train_NLL = [Get_NLL(
                              np.array(X_t[ i * batch_size: (i + 1) * batch_size].reshape(layer_input_size), dtype='float32')/np.array(X_t[ i * batch_size: (i + 1) * batch_size] , dtype='float32').max(),  
                              np.array(y_t[   i * batch_size: (i + 1) * batch_size] , dtype='int32') 
                                     ) 
                              for i in xrange(n_train_batches)
                        ]      
                    
        this_train_loss = np.mean(train_loss)
        
 

	  print this_train_loss
        print 'NLL ...... ' , np.mean(train_NLL ), np.mean(test_NLL)
        Train_list_scores.append(float(this_train_loss))
 



        if (epoch+1)%50 ==0 or epoch==70:
	      plt.figure(1)
	      plt.plot(np.arange(len(Test_list_scores)), np.array(Test_list_scores) , label= 'test error')
	      plt.hold(True)
	      plt.plot(np.arange(len(Train_list_scores)), np.array(Train_list_scores) , label= 'train error')
	      plt.legend(loc='upper left')
	      plt.hold(False)
	      plt.savefig('./MSE_test' +str(epoch)+'.png')
 
    return 
CCD()
