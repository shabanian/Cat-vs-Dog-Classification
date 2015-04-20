import numpy as np
 
import theano.tensor.signal.conv as Conv
 
import  scipy
import  scipy.signal
 
import scipy
import scipy.ndimage
 
import theano.tensor.nnet.conv as conv
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy
import pylab

from pylab import *
 
import theano
import theano.tensor as T
from MNIST import MNIST
import Image
from logistic_sgd import LogisticRegression, load_data
def dispims(M, height, width, border=0, bordercolor=0.0, layout=None,  gray = None,  name='no_name'):
    numimages = M.shape[1]
    if layout is None:
        n0 = int(np.ceil(np.sqrt(numimages)))
        n1 =  int(np.ceil(np.sqrt(numimages)))
    else:
        n0, n1 = layout
    im = bordercolor * np.ones(((height+border)*n0+border,(width+border)*n1+border),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[i*(height+border)+border:(i+1)*(height+border)+border,
                   j*(width+border)+border :(j+1)*(width+border)+border] = np.vstack((
                            np.hstack((np.reshape(M[:,i*n1+j],(height, width)),
                                   bordercolor*np.ones((height,border),dtype=float))),
                            bordercolor*np.ones((border,width+border),dtype=float)

                            ))
    if gray == None:
         
        pylab.imsave(arr = im, fname='./PSD_'+ name +'.png', cmap=pylab.cm.gray)
 
  
    else:
 
        pylab.savefig('sparse.png')

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
                     
                     
        self.input = input
    
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

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


 
def PSD( train_set_x, train_set_y, test_set_x, test_set_y, cost_type = 1, Slack =1.0):        
        rng = np.random.RandomState(23455)
        index = T.iscalar()
        x = T.matrix('x')
                                     
        batch_size = 200
        m = 28
        n = 28
        n_out = 256
    
        layer0_input = x.reshape((batch_size,  m*n))
        layer0 = HiddenLayer( rng,  layer0_input, n*m, n_out, None, None,
         activation=T.tanh)
 
        
        layer1_input = layer0.output 
        layer1 = HiddenLayer( rng, layer1_input,  n_out, n_out,None, None,
         activation=None)
 
        F = layer1.output 
                                 
        
        B_values = numpy.asarray(
                rng.uniform(
                low= -0.1,
                high= 0.1,
                size=(n*m, n_out)
                ),
                dtype=theano.config.floatX
        )
 
         
        B = theano.shared(value=B_values, name='B', borrow=True)
        
        Z_values = numpy.asarray(
                rng.uniform( 
                low= -0.1,
                high= 0.1,
                size=(batch_size,  n_out )
                ),
                dtype=theano.config.floatX
        )
        Z = theano.shared(value=Z_values, name='Z', borrow=True)
        ############################################################
        Lambda = T.fscalar()
        
        F_norm = T.sum((Z-F)**2, axis=1)
         
         
        if cost_type  ==1:
                cost = T.mean( T.sum(
                                (x-T.dot( B, Z.transpose()).transpose())**2 , axis=1),
                                axis=0) +Lambda*T.mean(
                                T.sum(abs(Z), axis =1),axis=0)+ T.mean( F_norm, axis=0) 
        if cost_type  == 2:           
               B_slack = theano.shared( Slack, name='B_slack ' )
               cost = T.mean( T.sum(
                                (x-T.dot( B, Z.transpose()).transpose())**2 , axis=1),
                                axis=0) +Lambda*T.mean(
                                T.sum(abs(Z), axis =1),axis=0)+ T.mean( F_norm, axis=0) + B_slack* T.sum(abs(abs(T.sum( B**2 , axis=0)) -1.0))
 
        
        Get_Train_Total_cost  = theano.function(
          [index, Lambda],
          cost,
          allow_input_downcast= True,
          givens={
          x: train_set_x[index * batch_size: (index + 1) * batch_size]
          }
          )
        Z_param =[Z]
        Z_grad = T.grad(cost, Z_param)
        L_rate_z = T.fscalar()
 
        Get_Z_grad  = theano.function(
                [index,  Lambda],
                Z_grad,
                allow_input_downcast= True,
                givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size]
                }
                )
        Z_updates = [
                (param_i, param_i - L_rate_z * grad_i)
                for param_i, grad_i in zip(Z_param, Z_grad)
                ]
        train_Esparse_model = theano.function(
                [index, L_rate_z, Lambda],
                cost,
                allow_input_downcast= True,
                updates = Z_updates,
                givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size]
                }
                )
        get_Z_grad = theano.function(
                [index, Lambda],
                Z_grad,
                allow_input_downcast= True,
                givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size]
                }
                )
    

        if cost_type  ==1:   
                U_d = [B]
                U_e =   [layer1.params[0]] + layer0.params
                 
                U_grads_d = T.grad( cost, U_d)
                U_grads_e = T.grad( cost, U_e)
                L_rate_slack  = T.fscalar() 
                L_rate_encoder  = T.fscalar()
                L_rate_decoder = T.fscalar()
                U_updates = [
                        (param_i,( param_i - L_rate_decoder  * grad_i)/( T.sqrt( T.sum(( param_i - L_rate_decoder  * grad_i)**2 , axis=0))))
                        for param_i, grad_i in zip(U_d, U_grads_d)
                        ] +  [
                        (param_i, param_i - L_rate_encoder * grad_i)
                        for param_i, grad_i in zip(U_e, U_grads_e)
                        ]
                train_ED_model = theano.function(
                        [index, L_rate_encoder, L_rate_decoder, Lambda],
                        cost,
                        updates=U_updates ,
                        allow_input_downcast= True,
                        givens={
                        x: train_set_x[index * batch_size: (index + 1) * batch_size]
                        }
                        ) 
        if cost_type  == 2:   
                U_d = [B] 
                U_e =   [layer1.params[0]] + layer0.params
                U_slack =[B_slack]
                U_grads_d = T.grad( cost, U_d)
                U_grads_e = T.grad( cost, U_e)
                U_grads_slack = T.grad( cost, U_slack)
                L_rate_encoder  = T.fscalar()
                L_rate_decoder = T.fscalar()
                L_rate_slack  = T.fscalar()
                U_updates = [
                        (param_i,  param_i - L_rate_decoder  * grad_i)
                        for param_i, grad_i in zip(U_d, U_grads_d)
                        ] +  [
                        (param_i, param_i - L_rate_encoder * grad_i)
                        for param_i, grad_i in zip(U_e, U_grads_e)
                        ]+  [
                        (param_i, param_i - L_rate_slack * grad_i)
                        for param_i, grad_i in zip(U_slack, U_grads_slack)
                        ]
                train_ED_model = theano.function(
                        [index, L_rate_encoder, L_rate_decoder, L_rate_slack , Lambda],
                        cost,
                        updates=U_updates ,
                        allow_input_downcast= True,
                        givens={
                        x: train_set_x[index * batch_size: (index + 1) * batch_size]
                        }
                        ) 
        
        get_F = theano.function(
            [index ],
            F,
            allow_input_downcast= True,
            givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
            }
            ) 

   
        y = T.ivector('y')
        Class_number = 10
        
        
        layer4_0_input = F#  
        layer4_0 = HiddenLayer( rng, layer4_0_input,  256, 100,None, None,
         activation=None)
   
        IN__PUT = layer4_0.output 
        
        
        layer4 = LogisticRegression(input=IN__PUT, n_in= 100, n_out= Class_number)
        print '... Logestic regression'
        params_Loges = layer4.params + U_e  
    
        cost_Loges  = layer4.negative_log_likelihood(y) 
        Accuracy_testSet_Loges  = theano.function(
                [index],
                   layer4.errors(y)  ,
                givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
                }
                )
        Accuracy_trainSet_Loges  = theano.function(
                [index],
                 layer4.errors(y),
                givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
                }
                )
        grads_Loges = T.grad(cost_Loges, params_Loges )
        learning_rate_Loges  = T.fscalar()
        updates_Loges = [
                (param_i, param_i - learning_rate_Loges  * grad_i)
                for param_i, grad_i in zip(params_Loges, grads_Loges)
                ]
        train_Loges_model = theano.function(
                [index, learning_rate_Loges ],
                cost_Loges,
                updates=updates_Loges,
                givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
                }
                )
        
        Error_testSet_Loges  = theano.function(
                [index],
                cost_Loges,
                givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
                }
                )
        Error_trainSet_Loges  = theano.function(
                [index],
                cost_Loges,
                givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
                } )
        RMSE_train = theano.function(
                [index],
                layer4.y_pred -y,
                givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
                }
                )
        RMSE_test = theano.function(
                [index],
                layer4.y_pred -y,
                givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]
                }
                )
        print '... Training'
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
  
         
  
         
        done_looping = False
        if cost_type  ==1:         
                learning_rate_E = np.float32(0.01)
                L_rate_e = 0.01
                L_rate_d = 0.01 
                learning_rate_E = 0.01
        if cost_type  == 2:         
                learning_rate_E = 0.01
                L_rate_e = 0.01
                L_rate_d = 0.01 
                L_rate_Slck = 0.0 
                  
        n_epochs = 30
         
         

        
        for lmbda in [10**(-7)]:
            print '... Lambda: ' , lmbda
            epoch =0
            Total_Esparse_cost_old =np.inf
            TAC_eq =0
      
            while (epoch < n_epochs) and (not done_looping):
                    print '.... epoch: ', epoch
                    epoch = epoch + 1
                    if epoch%5000 == 0:
                            L_rate_e = 0.1 * L_rate_e
                            L_rate_d = 0.1 * L_rate_d
                    
                    Total_Esparse_cost = 0.0
                    Total__avg_cost = 0.0
       
                    for minibatch_index in xrange(n_train_batches):
                
                            
                            Esparse_cost_old = np.inf
                            for star in range(10):
                                    Esparse_cost = np.mean( train_Esparse_model(minibatch_index ,learning_rate_E,lmbda  ))
                                    
                                    if np.sqrt(np.sum(np.array(get_Z_grad(minibatch_index, lmbda))**2))<0.001:  
                                             
                                            break
                                   
                            Total_Esparse_cost += Esparse_cost
        
        
        
        
                            if cost_type  == 1: 
                                    minibatch_avg_cost =  train_ED_model(minibatch_index, L_rate_e , L_rate_d, lmbda)
                            
                            if cost_type  == 2: 
                                    minibatch_avg_cost =  train_ED_model(minibatch_index, L_rate_e , L_rate_d, L_rate_Slck, lmbda)
                            
                                                       
                            Total__avg_cost += minibatch_avg_cost
                    
                    print'minimization cost' ,  Total_Esparse_cost/float(n_train_batches)
                    print Total__avg_cost/float(n_train_batches)
                    if Total_Esparse_cost<Total_Esparse_cost_old:
                            Total_Esparse_cost_old = Total_Esparse_cost
                    else:
                            TAC_eq+=1
                            if TAC_eq == 10:
                                    break
                  
        return 
                     
batch_size = 200
m = 28
n = 28
n_out = 256

mn = MNIST()
if mn.test():

        print('Loading from Mnist file')

x_train = []
t_train = []
Train_Number  = 60000
List = np.arange(Train_Number)
np.random.shuffle(List)
 
for i in List:
        
        x_train.append(np.array(mn._train_img[i], dtype='float32'))#/Max)
        t_train.append(int(mn._train_label[i] ))

x_train = np.array(x_train, dtype='float32') 
x_train /= np.max(x_train,axis=1).reshape((x_train.shape[0],1))*1.0
dispims(x_train[:100].transpose(), 28, 28, 1, layout=(10,10),
                            name='data')
t_train = np.array(t_train, dtype='int32') 
 
x_test = []
t_test = []
Test_Number  = 10000
List = np.arange( Test_Number )
np.random.shuffle(List)
for i in range(Test_Number ):
                x_test.append(np.array(mn._test_img[i], dtype='float32'))#/Max)
                t_test.append(int(mn._test_label[i]))
 
t_test = np.array(t_test, dtype='int32')

x_test = np.array(x_test, dtype='float32')
x_test /= np.max(x_test,axis=1).reshape((x_test.shape[0],1))*1.0
 
train_set_x= theano.shared(x_train)
train_set_y= theano.shared(t_train)

test_set_y= theano.shared(np.array(t_test, dtype='int32'))
test_set_x= theano.shared(x_test)

 
L1 = PSD( train_set_x, train_set_y, test_set_x, test_set_y, cost_type = 1)  
L2 = PSD( train_set_x, train_set_y, test_set_x, test_set_y, cost_type = 2,  Slack =1.0)
 
plt.figure(6)
plt.plot( np.arange(len(np.array( L1))) ,L1 , label= 'Test old')
plt.legend(loc='upper right')
plt.hold(True)
plt.plot(  np.arange(len(np.array( L2))) ,L2, label= 'Test new')
plt.legend(loc='upper right')
plt.hold(False)
plt.savefig('./MSE.png')  
