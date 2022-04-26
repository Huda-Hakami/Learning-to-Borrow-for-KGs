
import numpy as np 
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import random

class DataSet():
	def __init__(self):
		pass
	# ----------------------------------------------------
	def Retrieve_LDPs(self):
		self.id2LDP={}
		self.LDP2id={}
		with open("benchmarks/FB15K237/relation2id.txt") as rel2id:
			for i,line in enumerate(rel2id):
				if i!=0:
					line=line.strip().split()
					rel=line[0]
					id_=int(line[1])
					if id_>=237: # means we have textual rel (not kg r)
						self.id2LDP[id_-237]=rel
		self.LDP2id={v:k for k,v in self.id2LDP.items()}
	# ----------------------------------------------------
	def Retrieve_Entities(self):
		self.id2Entity={}
		self.Entity2id={}
		with open("benchmarks/FB15K237/entity2id.txt") as entit2id:
			for i,line in enumerate(entit2id):
				if i!=0:
					line=line.strip().split()
					entity=line[0]
					id_=int(line[1])
					self.id2Entity[id_]=entity
		self.Entity2id={v:k for k,v in self.id2Entity.items()}
	# ----------------------------------------------------
	def Retrieve_Entity_Embeddings(self):
		embeddings=np.load("pre-trained-KGEs/RelWalk_Embeddings.npy",allow_pickle=True,encoding = 'bytes').item()
		self.Entity_Embeddings=embeddings[b'ent_embeddings']
		# dimensionality of entity embeddings
		d=self.Entity_Embeddings.shape[1]
		print ("Entity embeddings of shape (num_of_entities,dim):", self.Entity_Embeddings.shape)
	# ----------------------------------------------------
	def Generate_LDP_Embeddings(self):
		model = SentenceTransformer('paraphrase-distilroberta-base-v1')
		LDPs=[]
		for i in range(len(self.id2LDP)):
			ldp=self.id2LDP[i].replace(":"," ")
			ldp=ldp.replace("[XXX]","subject")
			ldp=ldp.replace("[YYY]","object")
			LDPs.append(ldp)
		self.LDP_embeddings=model.encode(LDPs)
		print ("Shape of LDP embeddings:",self.LDP_embeddings.shape)
		normalized_LDPRep=normalize(self.LDP_embeddings)
		cosines=np.dot(normalized_LDPRep,normalized_LDPRep.T)
		print (cosines.shape)
		cosines = (cosines+1.0)/2.0

		file=open("similarities_between_ldps2.txt",'w')
		for i, ldp in enumerate(LDPs):
			file.write("##################\n")
			file.write("ldp: %s\n"%ldp)
			res=(-cosines[i]).argsort()
			top=res[:5]
			bottom=res[-5:]
			for ind in top:
				file.write("%s %s %f\n"%(ldp,LDPs[ind],cosines[i,ind]))
			file.write("----------------\n")
			for ind in bottom:
				file.write("%s %s %f\n"%(ldp,LDPs[ind],cosines[i,ind]))
		file.close()
	# ----------------------------------------------------
	def Retrieve_Training_Instances(self):
		self.PositiveExamples=[]
		self.Pos_pair2ldp={}
		self.Train_Entity_Pairs=set()
		
		with open("Positive_Examples.txt") as positives:
			for line in positives: 
				line=line.strip().split()
				h,t,ldp=int(line[0]),int(line[1]),int(line[2])
				ldp=ldp-237
				self.Pos_pair2ldp.setdefault((h,t),set())
				self.Pos_pair2ldp[(h,t)].add(ldp)
				self.Train_Entity_Pairs.add((h,t))
				self.PositiveExamples.append((h,t,ldp,1.0))
		self.PositiveExamples=list(set(self.PositiveExamples))
		self.NegativeExamples=[]
		self.Neg_pair2ldp={}
		print ("reading negative examples...")
		with open("Negative_Examples.txt") as negatives:
			for line in negatives: 
				line=line.strip().split()
				h,t,ldp=int(line[0]),int(line[1]),int(line[2])
				ldp=ldp-237
				self.Neg_pair2ldp.setdefault((h,t),set())
				self.Neg_pair2ldp[(h,t)].add(ldp)

		without_neg=set()
		for (h,t) in self.Pos_pair2ldp:
			if (h,t) not in self.Neg_pair2ldp:
				without_neg.add((h,t))
		for (h,t) in without_neg:
			del self.Pos_pair2ldp[(h,t)]
			self.Train_Entity_Pairs.remove((h,t))
		self.Train_Entity_Pairs=list(set(self.Train_Entity_Pairs))

		print ("Number of training entity-pairs:",len(self.Pos_pair2ldp))

		number_of_neg_examples=sum([len(self.Neg_pair2ldp[(h,t)]) for (h,t) in self.Neg_pair2ldp])
		print ("number of negative examples:",number_of_neg_examples)

		avg_neg_examples=np.mean([len(self.Neg_pair2ldp[(h,t)]) for (h,t) in self.Neg_pair2ldp])
		max_neg_examples=np.max([len(self.Neg_pair2ldp[(h,t)]) for (h,t) in self.Neg_pair2ldp])
		std_neg_examples=np.std([len(self.Neg_pair2ldp[(h,t)]) for (h,t) in self.Neg_pair2ldp])
		print ("Average Negative examples per entity-pair:",avg_neg_examples)
		print ("Maximum Negative examples per entity-pair:",max_neg_examples)
		print ("std Negative examples per entity-pair:",std_neg_examples)

		avg_pos_examples=np.mean([len(self.Pos_pair2ldp[(h,t)]) for (h,t) in self.Pos_pair2ldp])
		max_pos_examples=np.max([len(self.Pos_pair2ldp[(h,t)]) for (h,t) in self.Pos_pair2ldp])
		std_pos_examples=np.std([len(self.Pos_pair2ldp[(h,t)]) for (h,t) in self.Pos_pair2ldp])
		print ("Average Positive examples per entity-pair:",avg_pos_examples)
		print ("Maximum Positive examples per entity-pair:",max_pos_examples)
		print ("std Positive examples per entity-pair:",std_pos_examples)
		
	# ----------------------------------------------------