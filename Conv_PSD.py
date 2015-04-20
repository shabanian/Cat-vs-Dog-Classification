import numpy as np
import theano.tensor.signal.conv as Conv
import  scipy
import  scipy.signal 
import scipy
import scipy.ndimage
import theano.sandbox.cuda.fftconv 
import cPickle 
import os 
import theano.tensor.nnet.conv as conv  
import numpy
import pylab

from sklearn import svm
from sklearn.svm import NuSVC
from pylab import *
 
import theano
import theano.tensor as T
from CD_DATA  import load_gray 
 
 
def create_avgpool_filter(num_input_channels, pool_shape):
    filters = numpy.zeros((num_input_channels,) * 2 + pool_shape, dtype='float32')
    for i in range(num_input_channels):
        filters[i, i, :, :] = 1.
    filters /= numpy.product(pool_shape)
    return filters
  
def ova_svm_cost(W, b, x, y1):
 
    margin = y1 * (T.dot(x, W) + b)
    cost = hinge(margin).mean(axis = 0).sum()
    return cost
def lecun_lcn(  X, kernel_size=7, threshold = 1e-4, use_divisor=False):
        filter_shape = (1, 1, kernel_size, kernel_size)
        filters = gaussian_filter(kernel_size).reshape(filter_shape)
        filters = theano.shared(np.array(asarray(filters, dtype='float32')), borrow=True)

        convout = theano.tensor.nnet.conv2d(X, filters=filters, filter_shape=filter_shape, 
                            border_mode='full')

         
        mid = int(floor(kernel_size/2.))
        new_X = X - convout[:,:,mid:-mid,mid:-mid]

        if use_divisor: 
            sum_sqr_XX = conv2d(T.sqr(T.abs_(X)), filters=filters, 
                                filter_shape=filter_shape, border_mode='full')

            denom = T.sqrt(sum_sqr_XX[:,:,mid:-mid,mid:-mid])
            per_img_mean = denom.mean(axis=[2,3])
            divisor = T.largest(per_img_mean.dimshuffle(0,1,'x','x'), denom)
            divisor = T.maximum(divisor, threshold)

            new_X /= divisor
        return new_X 
def gaussian_filter(kernel_shape):

    x = zeros((kernel_shape, kernel_shape), dtype='float32')

    def gauss(x, y, sigma=1.591):
        Z = 2 * pi * sigma**2
        return  1./Z * exp(-(x**2 + y**2) / (2. * sigma**2))

    mid = floor(kernel_shape/ 2.)
    for i in xrange(0,kernel_shape):
        for j in xrange(0,kernel_shape):
            x[i,j] = gauss(i-mid, j-mid)

    return x / sum(x)
 
def gradient_updates_momentum(cost, params, learning_rate, momentum):
    assert momentum < 1 and momentum >= 0
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    return updates
def ova_prediction(W, b, x):
 
    return T.argmax(T.dot(x, W) + b, axis = 1)
 
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
 





def relu(x):
    return x * (x > 1E-6)




def PSD_conv_linear_combination():
    
 
          
        Decoder_size =(1, 64, 11,11) 
        Encoder_size = ( 64,1,11,11) 
      
        phi = T.tanh
        Data_type = 1
	    

        if Data_type == 1:

 
		print "... Loading Cat_vs_Dog data"
		size =70
		m = size
		n = size
		Input_size = [size,size] 
	 
		
		train, valid, test = load_gray()  
		x_test =test[0].astype('float32')  
		y_test = test[1].astype('int32') 
		x_valid = valid[0].astype('float32') 
		y_valid = valid[1].astype('int32')  
		x_train = train[0].astype('float32') 
		y_train = train[1].astype('int32')  	  
 	  
 
	        
		meanstd_train = x_train.std()
		mean = x_train.mean(1).reshape(( x_train.shape[0],1))
		var = x_train.std(1).reshape(( x_train.shape[0],1))+ 0.1 * meanstd_train
		mean2 = x_test.mean(1).reshape(( x_test.shape[0],1))
		meanstd_test = x_test.std()
		var2 = x_test.std(1).reshape(( x_test.shape[0],1)) + 0.1 * meanstd_test 
		x_train -=  mean 
		x_train /= var
		
		x_test -= mean2
		x_test /= var2
 
		train_set_x= theano.shared(np.array(x_train.reshape((x_train.shape[0],1,size,size)), dtype='float32'))
		train_set_y= theano.shared(np.array(y_train, dtype='int32'))

		test_set_y= theano.shared(np.array(y_test, dtype='int32'))
		test_set_x= theano.shared(np.array(x_test.reshape((x_test.shape[0],1,size,size)), dtype='float32'))  
	   
		
		train_set_x = T.reshape(lecun_lcn(train_set_x, kernel_size=7, threshold = 1e-4, use_divisor=False),
		          ((x_train.shape[0],size*size)))
		test_set_x = T.reshape(lecun_lcn(	test_set_x, kernel_size=7, threshold = 1e-4, use_divisor=False), 
		           ((x_test.shape[0],size*size)))

		train_set_x /= T.reshape( T.max(train_set_x,axis=1)*1.0,(x_train.shape[0],1))
		test_set_x /=T.reshape(  T.max(test_set_x , axis=1)*1.0,(x_test.shape[0],1))
 
		dispims(np.array(f()).reshape((x_train.shape[0], size*size))[:100].transpose(), size,size, 0, layout=(10,10),
				    name='data.png' ) 
 
 
	 
        batch_size = 200


        Lambda = 10.0**(-3)
        w = Input_size[0]
        s = Decoder_size[2]
        h = Input_size[1]
        M = Decoder_size[1]
        Sparse_Matrix_size = ( batch_size, M, w+s-1,h+s-1)
        rng = np.random.RandomState()
        index = T.lscalar()
        x = T.cast(T.fmatrix('x'), 'float32') 
        y = T.ivector('y')
        I = T.cast(x.reshape((batch_size, Input_size[0],Input_size[1])), 'float32')
        I  = I.dimshuffle(0,'x',1, 2)
 

        fan_in = np.prod(Decoder_size[1:])
 
        fan_out = (Decoder_size[0] * np.prod(Decoder_size[2:]) )/(np.prod((2.,2.)))
 
        D_bound = 0.01 
        Decoder_Matrix =theano.shared( np.asarray(
                                rng.uniform(low=-D_bound,
                                high=D_bound,
                                size = Decoder_size),
                                dtype='float32'), borrow=True) 
      
        En_bound = 0.01
        Encoder_Matrix =theano.shared( np.asarray(
                                rng.uniform(low=-D_bound,
                                high=En_bound,
                                size = Encoder_size),
                                dtype='float32'), borrow=True)
 
        E_bound = 0.01  
        Esparse_Matrix = theano.shared( np.asarray(
                                rng.uniform(low=-E_bound,
                                high=E_bound,
                                size =Sparse_Matrix_size ),
                                dtype='float32' ), borrow=True)
 
        P1 =T.reshape(  theano.sandbox.cuda.dnn.dnn_conv(
                         Esparse_Matrix, 
                        Decoder_Matrix),(batch_size, Input_size[0],Input_size[1])) 
 

        Sum = T.reshape(T.mean(P1, axis=0),(1, Input_size[0],Input_size[1]))
 
        E1 =0.5*T.sum(T.mean(( I-Sum)**2, axis=0))
 
        Encoder_mapp = T.reshape (T.tanh(  theano.sandbox.cuda.dnn.dnn_conv(
            	I, 
            	  Encoder_Matrix,'full' )),Sparse_Matrix_size  ) 
	 
 	 
        Reconstruction =     theano.sandbox.cuda.dnn.dnn_conv(
            	img= Encoder_mapp,
            	kerns=   Decoder_Matrix )  
        


 

	get_Reconstruction = theano.function (
                     [index],
                      Reconstruction,
		    givens={
		    x:   train_set_x[index * batch_size: (index + 1) * batch_size]
		    }
		    )

 
        P2 = T.sum(T.mean( ( Esparse_Matrix  -Encoder_mapp  )**2,  axis=0)  )
  
 
        cost = E1 +    P2  + Lambda*T.sum(T.mean(abs(Esparse_Matrix), axis =0))
  
        test_model = theano.function(
                        [index],
                        cost,
                        givens={
                        x:  test_set_x[index * batch_size: (index + 1) * batch_size]
                        }
                        )
        test_traint_model = theano.function(
                  [index],
                  cost,
                  givens={
                  x:  train_set_x[index * batch_size: (index + 1) * batch_size]
                  }
                  )
        Esparse_Matrix_grad = T.grad(cost,  [Esparse_Matrix])
   
        Esparse_lr =  T.fscalar()
        Esparse_Update  = gradient_updates_momentum(cost, params =[Esparse_Matrix] , learning_rate = Esparse_lr, momentum=0.9)
 
 
        train_Esparse_model = theano.function(
                [index, Esparse_lr ],
                cost,
                updates= Esparse_Update,
                givens={
                x:  train_set_x[index * batch_size: (index + 1) * batch_size]
                }
                )
 

        get_Z_grad = theano.function(
		[index],
		 Esparse_Matrix_grad ,
		givens={
		x:   train_set_x[index * batch_size: (index + 1) * batch_size]
		})
 
	U_d = [  Decoder_Matrix]
	U_e =   [ Encoder_Matrix]  
	
	U_grads_d = T.grad( cost, U_d)
	U_grads_e = T.grad( cost, U_e)
	
	L_rate_encoder  = T.fscalar()
	L_rate_decoder = T.fscalar()
	
	U_updates = [
		(param_i,( param_i - L_rate_decoder  * grad_i)/(1E-3 + T.sqrt( T.sum(( param_i - L_rate_decoder  * grad_i)**2 , axis=0))))
		for param_i, grad_i in zip(U_d, U_grads_d)
		] +   [
		(param_i, param_i - L_rate_encoder * grad_i)
		for param_i, grad_i in zip(U_e, U_grads_e)
		] 
        
        
 
        train_ED_model = theano.function(
                [index, L_rate_encoder,L_rate_decoder],
                cost,
                updates=U_updates,
                givens={
                x:   train_set_x[index * batch_size: (index + 1) * batch_size]
                }
                )
 
  
 
        F = T.reshape( abs(Encoder_mapp), (batch_size, M*(( w+s-1)**2)))
        meanstd_F = F.std()
	  F -=  F.mean(1)[:,None]
	  F /= F.std(1)[False:,None]+ 0.1 * meanstd_F
 
	 
	F = T.reshape( abs(Encoder_mapp),  (batch_size,M, w+s-1, w+s-1)) 
	con_out  = theano.sandbox.cuda.dnn.dnn_pool(img= F, ws= (2,2),    mode='average' )
 
        pool_out     = T.reshape(con_out, (batch_size, M* (( w+s-1)/2)**2))
        get_pool_out_train = theano.function (
                     [ index ],
                     pool_out,
                     givens={
		     x:    train_set_x[index * batch_size: (index + 1) * batch_size]
		    }
                     )
        get_pool_out_test = theano.function (
                     [ index ],
                     pool_out,
                     givens={
		     x: test_set_x[index * batch_size: (index + 1) * batch_size]
		      }
                     )                 
 
 
 
 
        print '... Training'
   
        n_train_batches = x_train.shape[0] / batch_size
   
	n_epochs = 100
	L_rate_e = 10.0**(-3) 
	L_rate_d = 10.0**(-3) 
	learning_rate_E = 10.0**(-2 ) 
      minibatch_avg_cost_total =0
 
      for e in xrange(n_epochs):
         
		    
                print '.... epoch: ',  e
           
                for minibatch_index in xrange(n_train_batches):
                            
                         
                        for star in range(10):
                                   
                                    Esparse_cost = train_Esparse_model(minibatch_index, learning_rate_E  )
                   
				              if np.sqrt(np.sum(np.array(get_Z_grad(minibatch_index))**2))<0.001: # or Esparse_cost_old <Esparse_cost:
					    
					                  break 
 
                        minibatch_avg_cost =  train_ED_model(minibatch_index,L_rate_e,L_rate_d)
                        minibatch_avg_cost_total += minibatch_avg_cost/float(n_train_batches)
		  
 
      construct =np.array( [ np.array(get_Reconstruction(i) ) for i in range(n_train_batches)]).reshape((n_train_batches*batch_size, Input_size[0]* Input_size[1])) [:100]
	 
      clipped_sample = np.maximum(construct , 0.0)
      clipped_sample = np.minimum(clipped_sample, 1.0)

 
      dispims(clipped_sample.transpose(), Input_size[0],Input_size[0], 0, layout=(10,10),
			    name='Conv_reConstruct')
 

 
      pool_out_train = np.array([  get_pool_out_train( i  )   for i in xrange(n_train_batches)]).reshape( (batch_size*n_train_batches, M* (( w+s-1)/2)**2))
	pool_out_test = np.array([get_pool_out_test( i  ) for i in xrange(n_test_batches)]).reshape( (batch_size*n_test_batches, M* (( w+s-1)/2)**2))
	 
	clf_linear = svm.LinearSVC(penalty='l2', loss='l2', dual=False, tol=0.00001, C=0.9, multi_class='ovr',
                            fit_intercept=False, intercept_scaling=1, class_weight=None, verbose=0, random_state=None)
	clf_linear.fit(pool_out_train,train_y)
	train_accuracy = clf_linear.score(pool_out_train  ,train_y)
	print 'Linear SVM Results'
	print train_accuracy
	test_accuracy = clf_linear.score(pool_out_test  ,test_y)
      print 'test accuracy: ', test_accuracy

  
PSD_conv_linear_combination()
 
 
