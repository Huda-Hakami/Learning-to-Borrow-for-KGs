import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

class RelRep():
	def __init__(self,activ):
		self.Learning_Rate=0.01
		self.momentum=0.9
		self.activ=activ
		pass
	#------------------------------------------------------------
	def Network_Model(self,Entity_Embeddings,LDP_Embeddings,BatchNorm,Num_Hidden_Layers,ldp_dim,Hidden_dim,l2_reg):
		# Hyperparameters
		self.NumHiddenLayers=Num_Hidden_Layers
		self.HiddenDim=Hidden_dim
		self.ldp_dim=ldp_dim
		self.BN=BatchNorm
		self.l2Reg=l2_reg


		self.EntityEmbeddings=tf.Variable(Entity_Embeddings,trainable=False,name='EntityEmbeddings')
		self.LDP_Embeddings=tf.Variable(LDP_Embeddings,trainable=False,name='LDPEmbeddings')
		# IDs Placeholders
		self.h_ids=tf.placeholder(tf.int32,[None])
		self.t_ids=tf.placeholder(tf.int32,[None])
		self.pos_ldp_e=tf.placeholder(tf.float32,[None,self.ldp_dim])
		self.neg_ldp_e=tf.placeholder(tf.float32,[None,self.ldp_dim])


		# Lookup for Embeddings
		h_e=tf.nn.embedding_lookup(self.EntityEmbeddings,self.h_ids)
		t_e=tf.nn.embedding_lookup(self.EntityEmbeddings,self.t_ids)
		

		self.pkeep=tf.placeholder(tf.float32,name="pkeep")
		self.is_training = tf.placeholder(tf.bool)

		# Prepare the MLP layers
		input_layer=tf.concat([h_e,t_e,tf.subtract(h_e,t_e),tf.multiply(h_e,t_e)],1)
		InputDim=int(input_layer.get_shape()[1])

		neurons=[InputDim]
		for i in range(self.NumHiddenLayers):
			neurons.append(self.HiddenDim)

		# Define the Parameters of the MLP
		self.hidden={}
		for layer in range(1,self.NumHiddenLayers+1):
			weight_shape=[neurons[layer-1],neurons[layer]]
			bias_shape=[neurons[layer]]
			self.hidden['W%d'%layer]=variable(weight_shape,'W%d'%layer)
			self.hidden['b%d'%layer]=variable(bias_shape,'b%d'%layer)

		# Perfrom the computation of the defined layers
		self.Last_output=self.Feed_pair_toMLP(input_layer)

		print ('last_output shape:',self.Last_output.get_shape())
		self.l2_regularizer=tf.nn.l2_loss(self.hidden['W1'])+tf.nn.l2_loss(self.hidden['b1'])
		for i in range(2,self.NumHiddenLayers+1):
			self.l2_regularizer+=tf.nn.l2_loss(self.hidden['W%d'%i])+tf.nn.l2_loss(self.hidden['b%d'%i])
	#------------------------------------------------------------
	def Feed_pair_toMLP(self,input_layer):
		output={}
		for layer in range(1,self.NumHiddenLayers+1):
			if layer==1:
				inp=input_layer
			else:
				inp=output['Y%d'%(layer-1)]
			# compute activations of the hidden layer
			if self.BN:
				output['Y%d'%layer]=self.hidden_layer_output_bn(inp,self.hidden['W%d'%layer])
			else:
				output['Y%d'%layer]=self.hidden_layer_output(inp,self.hidden['W%d'%layer],self.hidden['b%d'%layer])
		return output['Y%d'%self.NumHiddenLayers]
	#------------------------------------------------------------------	
	def hidden_layer_output(self,X,W,b):
		if self.activ=='sigmoid':
			return tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.matmul(X,W),b)),self.pkeep)
		elif self.activ=='tanh':
			return tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(X,W),b)),self.pkeep)
		elif self.activ=='relu':
			return tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(X,W),b)),self.pkeep)
		elif self.activ=='linear':
			return tf.nn.dropout(tf.add(tf.matmul(X,W),b),self.pkeep)
		else:
			raise ValueError
	#------------------------------------------------------------------
	def hidden_layer_output_bn(self,X,W):
		z=tf.matmul(X,W)
		z_bn=tf.layers.batch_normalization(z,training=self.is_training)

		if self.activ=='tanh':
			return tf.nn.tanh(z_bn)
		elif self.activ=='relu':
			return tf.nn.relu(z_bn)
		elif self.activ=='sigmoid':
			return tf.nn.sigmoid(z_bn)
		else:
			raise ValueError
	#------------------------------------------------------------
	def define_loss(self):
		dot_loss=tf.reduce_sum(tf.multiply(self.Last_output,tf.subtract(self.neg_ldp_e,self.pos_ldp_e)),1)
		margin = tf.constant(1.) 
		self.loss = tf.reduce_mean(tf.nn.relu(margin+dot_loss)) + (self.l2Reg * self.l2_regularizer)
#------------------------------------------------------------
	def optimize(self):
		batch = tf.Variable(0, trainable=False)
		# optimizer=tf.train.AdagradOptimizer(self.Learning_Rate)
		optimizer=tf.train.MomentumOptimizer(self.Learning_Rate,self.momentum)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.train_step=optimizer.minimize(self.loss,global_step=batch)
#------------------------------------------------------------------
#  ============ End of the RelRep class ============
def variable(shape,var_name):
		return tf.Variable(tf.random_uniform(shape,minval=-1,maxval=1,dtype=tf.float32),name=var_name)
# -----------------------------------------------------------
