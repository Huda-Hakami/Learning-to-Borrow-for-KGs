from dataset import DataSet
import SuperBorrowModel
import random
import numpy as np
from sklearn.preprocessing import normalize
from algebra import normalize, cosine
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

class Training():
	def __init__(self):
		# Hyperparameters
		HL=2
		self.ldp_dim=768
		Hdim=768
		BN=True #boolean variable T/F for batch normalization on MLP 
		l2_reg=0.001 # L2 regularization coefficient
		self.pkeep=1.0 # 1.0 means no Dropout applied during training on the MLP
		activ='tanh'

		# Create relational model instance
		self.RelModel=SuperBorrowModel.RelRep(activ)
		self.RelModel.Network_Model(DS.Entity_Embeddings,DS.LDP_embeddings,BN,HL,self.ldp_dim,Hdim,l2_reg)
		self.RelModel.define_loss()
		self.RelModel.optimize()
		self.sess=tf.Session()
		self.sess.run(tf.global_variables_initializer())
		pass
	#------------------------------------------------------------------
	def Train_Model(self):
		epochs=50
		batchSize=128
		hist_loss=[]
		winn_loss=1e7

		Train,Valid=split_trainvalid(DS.Train_Entity_Pairs)
		
		for epoch in range(epochs):
			res=0.0
			MR,filt_hit1,filt_hit10,filt_hit30,filt_MRR,filt_MR=self.mesure_valid_performance(Valid)
			
			random.shuffle(Train)

			for minibatch in next_batch(batchSize,Train):
				pos_ldp_embeddings=self.prepare_ldp_embeddings(minibatch,"pos")
				neg_ldp_embeddings=self.prepare_ldp_embeddings(minibatch,"neg")
				
				h_ids,t_ids=shred_tuples(minibatch)

				train_data={self.RelModel.h_ids:h_ids,self.RelModel.t_ids:t_ids,self.RelModel.pos_ldp_e:pos_ldp_embeddings,\
							self.RelModel.neg_ldp_e:neg_ldp_embeddings,self.RelModel.pkeep:self.pkeep,self.RelModel.is_training:True}

				loss,_=self.sess.run([self.RelModel.loss,self.RelModel.train_step],feed_dict=train_data)
				res+=loss
			print ("new epoch %d, train_loss=%f, valid_acc=%f, Hit@1=%f, Hit@10=%f, Hit@30=%f, filt_MRR=%f, filt_MR=%f"%(epoch,res,MR,filt_hit1,filt_hit10,filt_hit30,filt_MRR,filt_MR))
		self.Borrow_LDPs()
	# -------------------------------------------------------
	def mesure_valid_performance(self,data):
		h_ids,t_ids=shred_tuples(data)
		valid_data={self.RelModel.h_ids:h_ids,self.RelModel.t_ids:t_ids,self.RelModel.pkeep:1.0,self.RelModel.is_training:False}
		RelRep=self.sess.run(self.RelModel.Last_output,feed_dict=valid_data)
		Ranks=[]
		filt_hit1=filt_hit10=filt_hit30=c=0
		filt_MRR=filt_MR=0.0
		

		DotProduct=np.dot(RelRep,DS.LDP_embeddings.T)

		for i,(h,t) in enumerate(data):
			res=(-DotProduct[i]).argsort()
			pos_ldps=[]
			for l in DS.Pos_pair2ldp[(h,t)]:
				rank=np.where(res==l)[0][0]+1
				pos_ldps.append(rank)
			min_rank=np.min(pos_ldps)
			Ranks.append(min_rank)
		for i, (h,t) in enumerate(data):
			res=(-DotProduct[i]).argsort()
			L=[(h,ldp,t) for ldp in DS.Pos_pair2ldp[(h,t)]]
			lis=[(h,id_,t) for id_ in res]
			for (h,ldp,t) in L: #For each correct instance (h,ldp,t)
				c+=1
				filt_rank=0
				for (h,ldp_,t) in lis:
					if (h,ldp_,t) not in L and (h,ldp_,t)!=(h,ldp,t):
						filt_rank+=1
					if (h,ldp_,t) == (h,ldp,t):
						filt_rank+=1
						break
				filt_MR += filt_rank
				filt_MRR += (1.0/filt_rank)
				filt_hit1 += float(filt_rank<=1)
				filt_hit10 += float(filt_rank<=10)
				filt_hit30 += float(filt_rank<=30)

		MR=np.mean(Ranks)
		filt_hit1 /= c
		filt_hit10 /= c
		filt_hit30 /= c
		filt_MRR /= c
		filt_MR /= c
		
		return MR,filt_hit1,filt_hit10,filt_hit30,filt_MRR,filt_MR
	# -------------------------------------------------------
	def prepare_ldp_embeddings(self,minibatch,flag):
		if flag=="pos":
			pair2ldp=DS.Pos_pair2ldp
		elif flag=="neg":
			pair2ldp={}
			for (h,t) in minibatch:
				pair2ldp[(h,t)]=random.sample(list(DS.Neg_pair2ldp[(h,t)]),1)
		ldp_embeddings=np.zeros((len(minibatch),self.ldp_dim))
		for i,(h,t) in enumerate(minibatch):
			x=np.zeros(self.ldp_dim)
			for ldp in pair2ldp[(h,t)]:
				x+=DS.LDP_embeddings[ldp]
			x=x/len(pair2ldp[(h,t)])
			ldp_embeddings[i]=x
		return ldp_embeddings
	# -------------------------------------------------------
	def Borrow_LDPs(self):
		without_mentions=set()
		with open("benchmarks/FB15K237/test2id_withoutMentions.txt") as without:
			for i,line in enumerate(without):
				if i!=0:
					h,t,r=line.strip().split()
					h,t=int(h),int(t)
					without_mentions.add((h,t))
		without_mentions=list(without_mentions)
		h_ids,t_ids=shred_tuples(without_mentions)
		test_data={self.RelModel.h_ids:h_ids,self.RelModel.t_ids:t_ids,self.RelModel.pkeep:1.0,self.RelModel.is_training:False}
		RelRep=self.sess.run(self.RelModel.Last_output,feed_dict=test_data)

		normalized_RelRep=normalize(RelRep)
		normalized_LDPRep=normalize(DS.LDP_embeddings)

		DotProduct=np.dot(RelRep,DS.LDP_embeddings.T)
		cosines=np.dot(normalized_RelRep,normalized_LDPRep.T)
		cosines = (cosines+1.0)/2.0

		n=[1,3,10,15,20,25,30]
		for number in n:
			borrowed_training_examples_fromDotProduct=set()
			for i,(h,t) in enumerate(without_mentions):
				res1=(-DotProduct[i]).argsort()[:number]
				for ind in res1:
					borrowed_training_examples_fromDotProduct.add((h,t,ind+237,DotProduct[i,ind]))
					
			file1=open("Borrowed_instances_dotproduct_thr%s.txt"%number,'w')
			file2=open("Borrowed_instances_dotproduct_thr%s_withScores.txt"%number,'w')

			for (h,t,ldp,score) in borrowed_training_examples_fromDotProduct:
				file1.write("%d %d %d\n"%(h,t,ldp))
				file2.write("%d %d %d %f\n"%(h,t,ldp,score))

			file1.close()
			file2.close()

#  ============ End of the Evaluation class ============
def next_batch(batchSize,data):
	# loop over our dataset in mini-batches of size `batchSize`
	for i in np.arange(0, len(data), batchSize):
		# yield the current batched data 
		yield data[i:i + batchSize]
#------------------------------------------------------------------
def shred_tuples(tuples):
	h_ids=[t[0] for t in tuples]
	t_ids=[t[1] for t in tuples]
	return h_ids,t_ids
# -------------------------------------------------------
def split_trainvalid(data):
	random.shuffle(data)
	valid = data[:int((len(data)*10)/100)]
	train = data[int((len(data)*10)/100):]
	print ("number of training entity-pairs",len(train),"number of valid entity-pairs",len(valid))
	exit()
	return train,valid
# -------------------------------------------------------
if __name__=="__main__":
	
	DS=DataSet()
	DS.Retrieve_LDPs()
	DS.Retrieve_Entities()
	DS.Retrieve_Entity_Embeddings()
	print ("number of LDPs=",len(DS.LDP2id))
	print ("number of Entities=",len(DS.id2Entity))

	DS.Generate_LDP_Embeddings()
	DS.Retrieve_Training_Instances()

	# Training & Evaluaiton
	Eval=Training()
	Eval.Train_Model()
