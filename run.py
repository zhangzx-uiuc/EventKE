import json
import numpy as np
from helper import *
from data_loader import *
import random

from torch.utils.data import TensorDataset

# sys.path.append('./')
from model.models import *

class Runner(object):

	def load_events(self):
		event_type_dir = "./data/" + self.p.dataset + "/event_types.json"
		event_ids_dir = "./data/" + self.p.dataset + "/event_ids.json"
		event_entity_dir = "./data/" + self.p.dataset + "/event2ent.json"

		self.role2id = {}
		self.event_edges = {} # {event_idx: [[ent_idx1, role_idx1], [ent_idx2, role_idx2], ...]}

		with open(event_type_dir, "r", encoding="utf-8") as f:
			self.event_types = json.loads(f.readline())
		
		with open(event_ids_dir, "r", encoding="utf-8") as f:
			self.evt2id = json.loads(f.readline())
		
		with open(event_entity_dir, "r", encoding="utf-8") as f:
			evt2ent = json.loads(f.readline())

		self.evt2ent = evt2ent

		# print(evt2ent)
		# read event2ent into role ids
		role_num = 0

		for event_id in evt2ent:
			args = evt2ent[event_id]["args"]
			for arg in args:
				role_type = args[arg]
				if role_type not in self.role2id:
					self.role2id.update({role_type: role_num})
					role_num += 1
		
		for event_id in evt2ent:
			args = evt2ent[event_id]["args"]
			event_idx = self.evt2id[event_id]

			for arg in args:
				role_type_idx = self.role2id[args[arg]]
				ent_idx = self.ent2id[arg]

				if event_idx not in self.event_edges:
					self.event_edges.update({event_idx: [[ent_idx, role_type_idx]]})
				else:
					self.event_edges[event_idx].append([ent_idx, role_type_idx].copy())
		
		role_num = len(self.role2id)
		self.p.role_num = role_num
		self.p.event_num = len(self.evt2id)
		max_role_num = max([len(evt2ent[key]["args"]) for key in evt2ent])
		self.p.max_role_num = max_role_num

		# print(self.event_edges)
		evt2ent_list = [[] for _ in range(self.p.event_num)]
		for evt_id in self.event_edges:
			for ent_id in self.event_edges[evt_id]:
				evt2ent_list[evt_id].append(ent_id[0])
		self.evt2ent_list = evt2ent_list
		# print(evt2ent_list)
		# print(evt2ent_list)
		# print(self.event_edges)
		# print(self.role2id)

	def load_data(self):
		"""
		Reading in raw triples and converts it into a standard format. 

		Parameters
		----------
		self.p.dataset:         Takes in the name of the dataset (FB15k-237)
		
		Returns
		-------
		self.ent2id:            Entity to unique identifier mapping
		self.id2rel:            Inverse mapping of self.ent2id
		self.rel2id:            Relation to unique identifier mapping
		self.num_ent:           Number of entities in the Knowledge graph
		self.num_rel:           Number of relations in the Knowledge graph
		self.embed_dim:         Embedding dimension used
		self.data['train']:     Stores the triples corresponding to training dataset
		self.data['valid']:     Stores the triples corresponding to validation dataset
		self.data['test']:      Stores the triples corresponding to test dataset
		self.data_iter:		The dataloader for different data splits

		"""

		ent_set, rel_set = OrderedSet(), OrderedSet()
		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = line.strip().split('\t')
				ent_set.add(sub)
				rel_set.add(rel)
				ent_set.add(obj)

		self.p.entity_embed_dir = "./data/" + self.p.dataset + "/entity_embed.npy"
		# print(self.p.entity_embed_dir)
		entity_ids_dir = "./data/" + self.p.dataset + "/entity_ids.json"
		entity_types_dir = "./data/" + self.p.dataset + "/entity_types.json"
		relation_ids_dir = "./data/" + self.p.dataset + "/relation_ids.json"

		with open(entity_ids_dir, "r", encoding="utf-8") as f:
			self.ent2id = json.loads(f.readline())
		
		with open(entity_types_dir, "r", encoding="utf-8") as f:
			entity_types = json.loads(f.readline())
		
		self.p.num_ent		= len(self.ent2id)
		self.p.ent_num		= len(self.ent2id)
		# load entity types to type_id:
		self.ent_type_to_id = {}
		entity_type_num = 0

		for ent_id in entity_types:
			if entity_types[ent_id] not in self.ent_type_to_id:
				self.ent_type_to_id.update({entity_types[ent_id]: entity_type_num})
				entity_type_num += 1

		self.id_to_ent_type = {idx: t for t, idx in self.ent_type_to_id.items()}
		self.p.entity_type_num = len(self.ent_type_to_id)

		train_ent_labels, test_ent_labels, valid_ent_labels = [], [], []
		train_ent_ids, test_ent_ids, valid_ent_ids = [], [], []
		
		train_num = int(self.p.ent_num * 0.8)
		valid_num = int(self.p.ent_num * 0.1) + 1
		test_num = self.p.ent_num - train_num - valid_num

		# print(train_num)

		for ent_id in self.ent2id:
			ent_idx = self.ent2id[ent_id]
			if ent_idx >=0 and ent_idx < train_num:
				train_ent_ids.append(ent_idx)
				train_ent_labels.append(self.ent_type_to_id[entity_types[ent_id]])
			elif ent_idx >= train_num and ent_idx < train_num + valid_num:
				valid_ent_ids.append(ent_idx)
				valid_ent_labels.append(self.ent_type_to_id[entity_types[ent_id]])
			else:
				test_ent_ids.append(ent_idx)
				test_ent_labels.append(self.ent_type_to_id[entity_types[ent_id]])
		
		self.train_ent_labels = torch.LongTensor(train_ent_labels).to(self.device)
		self.train_ent_ids = torch.LongTensor(train_ent_ids).to(self.device)

		self.valid_ent_labels = torch.LongTensor(valid_ent_labels).to(self.device)
		self.valid_ent_ids = torch.LongTensor(valid_ent_ids).to(self.device)

		self.test_ent_labels = torch.LongTensor(test_ent_labels).to(self.device)
		self.test_ent_ids = torch.LongTensor(test_ent_ids).to(self.device)

		
		with open(relation_ids_dir, "r", encoding="utf-8") as f:
			self.rel2id = json.loads(f.readline())

		# self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		# self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}

		rel2id_dict = self.rel2id.copy()
		for key in rel2id_dict:
			self.rel2id.update({key+'_reverse': rel2id_dict[key]+len(rel2id_dict)})

		# print(self.rel2id)
		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

		''' 
		ent2id: {entity_id: 0}
		rel2id: {rel_id: 0, rel_id_reverse: 1}
		'''

		# also load relation classification dataset
		train_rel_labels, train_rel_start, train_rel_end = [], [], []
		valid_rel_labels, valid_rel_start, valid_rel_end = [], [], []
		test_rel_labels, test_rel_start, test_rel_end = [], [], []

		self.p.num_rel = len(self.rel2id) // 2 

		for line in open('./data/{}/train.txt'.format(self.p.dataset)):
			sub, rel, obj = line.strip().split('\t')
			start_idx = self.ent2id[sub]
			train_rel_start.append(start_idx)
			end_idx = self.ent2id[obj]
			train_rel_end.append(end_idx)
			rel_idx = self.rel2id[rel]
			train_rel_labels.append(rel_idx)
		
		self.train_rel_labels = torch.LongTensor([train_rel_labels]).to(self.device).t()
		self.train_rel_start = torch.LongTensor([train_rel_start]).to(self.device).t()
		self.train_rel_end = torch.LongTensor([train_rel_end]).to(self.device).t()

		for line in open('./data/{}/valid.txt'.format(self.p.dataset)):
			sub, rel, obj = line.strip().split('\t')
			start_idx = self.ent2id[sub]
			valid_rel_start.append(start_idx)
			end_idx = self.ent2id[obj]
			valid_rel_end.append(end_idx)
			rel_idx = self.rel2id[rel]
			valid_rel_labels.append(rel_idx)
		
		self.valid_rel_labels = torch.LongTensor([valid_rel_labels]).to(self.device).t()
		self.valid_rel_start = torch.LongTensor([valid_rel_start]).to(self.device).t()
		self.valid_rel_end = torch.LongTensor([valid_rel_end]).to(self.device).t()

		for line in open('./data/{}/test.txt'.format(self.p.dataset)):
			sub, rel, obj = line.strip().split('\t')
			start_idx = self.ent2id[sub]
			test_rel_start.append(start_idx)
			end_idx = self.ent2id[obj]
			test_rel_end.append(end_idx)
			rel_idx = self.rel2id[rel]
			test_rel_labels.append(rel_idx)
		
		self.test_rel_labels = torch.LongTensor([test_rel_labels]).to(self.device).t()
		self.test_rel_start = torch.LongTensor([test_rel_start]).to(self.device).t()
		self.test_rel_end = torch.LongTensor([test_rel_end]).to(self.device).t()

		# 这里的num_rel的含义是relation type的数量 而不是relation edge的数量，其实edge的数量就等于triples的数量 
		# num_rel在这里指的是：没有添加reverse边的时候的数量
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

		self.data = ddict(list)
		sr2o = ddict(set)

		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = line.strip().split('\t')
				sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
				self.data[split].append((sub, rel, obj))

				if split == 'train': 
					sr2o[(sub, rel)].add(obj)
					sr2o[(obj, rel+self.p.num_rel)].add(sub)
		# sr2o的格式：dict, {(h, r): Set({t1, t2, ...})}
		# for key in sr2o:
		# 	if len(sr2o[key]) > 1:
		# 		print(key)
		# 		print(sr2o[key])
		# 		print('\n')
		# print(sr2o)
		self.data = dict(self.data)
		# print(self.data)
		# self.data: {"train": [(h, r, t), (h, r, t)], "valid": [(h, r, t), (h, r, t)]}
		self.sr2o = {k: list(v) for k, v in sr2o.items()} # only contains training sr2os
		# self.sr2o的格式：dict, {(h, r): [t1, t2, ...])}
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)
				sr2o[(obj, rel+self.p.num_rel)].add(sub)

		self.sr2o_all = {k: list(v) for k, v in sr2o.items()} # contains training, validataion, and testing sr2os
		# self.sr2o_all的格式：dict, {(h, r): [t1, t2, ...])}
		# print(type(self.sr2o))
		# print(type(self.sr2o[(6593,9)]))
		self.triples  = ddict(list)

		for (sub, rel), obj in self.sr2o.items():
			self.triples['train'].append({'triple':(sub, rel, -1), 'obj': obj, 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

		# self.triples['train']: {'triple': (h, r, -1), 'label': [t1, t2, t3,...], 'sub_samp': 1}
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				rel_inv = rel + self.p.num_rel
				self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		# 注意：self.triples["train"] 里面只包含了训练集的所有triples，"triple"的最后一个值是-1，但是"valid", "test"里面的label包含了所有的可能的labels，(包括训练集和测试集)

		self.triples = dict(self.triples)

		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		self.data_iter = {
			'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size),
			'valid_head':   get_data_loader(TestDataset,  'valid_head', self.p.batch_size, False),
			'valid_tail':   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size, False),
			'test_head':   	get_data_loader(TestDataset,  'test_head',  self.p.batch_size, False),
			'test_tail':   	get_data_loader(TestDataset,  'test_tail',  self.p.batch_size, False),
		}

		self.load_events()
		self.id2evt = {idx: evt for evt, idx in self.evt2id.items()}

		self.edge_index, self.edge_type = self.construct_adj()
		# print(self.edge_type.shape)
		# print(self.edge_index.shape)
		self.event_edge_index, self.event_index, self.role_type, self.role_mask, self.entity_event_index, self.entity_mask = self.construct_event_adj()
		# print(self.entity_event_index.shape)

	def construct_adj(self):
		"""
		Constructor of the runner class

		Parameters
		----------
		
		Returns
		-------
		Constructs the adjacency matrix for GCN
		
		"""
		# edge_index: [] -- a list of tuples: [(subj1, obj1), (subj2, obj2), ...]
		# edge_type: [] -- a list of edge_type_idxs: [r1,r2,...]
		edge_index, edge_type = [], []

		# self.data[split]: list of (h,r,t)
		for sub, rel, obj in self.data['train']:
			edge_index.append((sub, obj))
			edge_type.append(rel)

		# Adding inverse edges
		for sub, rel, obj in self.data['train']:
			edge_index.append((obj, sub))
			edge_type.append(rel + self.p.num_rel)
		# print(self.device)
		edge_list = [[] for _ in range(self.p.ent_num)]
		for start, end in edge_index:
			edge_list[end].append(start)
		# print(edge_index)
		self.edge_list = edge_list
		# print(edge_list)
		# edge_index	= torch.LongTensor(edge_index).to(self.device).t()
		edge_index	= torch.LongTensor(edge_index).to(self.device).t()
		edge_type	= torch.LongTensor(edge_type). to(self.device)
		
		# edge_index: torch.LongTensor(2, 2*edge_num)
		# edge_type: torch.LongTensor(2*edge_num)
		# print(edge_type.shape)
		# print(edge_index.shape)
		# construct a tensor for all 1-hop previous neighbors
		


		return edge_index, edge_type
	
	def construct_event_adj(self):
		event_index = torch.zeros(self.p.event_num, self.p.max_role_num, dtype=torch.long).to(self.device)
		role_type = torch.zeros(self.p.event_num, self.p.max_role_num, dtype=torch.long).to(self.device)
		role_mask = torch.zeros(self.p.event_num, self.p.max_role_num).to(self.device)

		# check the maximum number of events
		ent2evt = {}
		for evt_idx in self.event_edges:
			args = self.event_edges[evt_idx]
			for arg in args:
				if arg[0] not in ent2evt:
					ent2evt.update({arg[0]: [evt_idx]})
				else:
					ent2evt[arg[0]].append(evt_idx)
		
		self.ent2evt = ent2evt
		# print(self.ent2evt)
		# 改写ent2evt成为list形式
		ent2evt_list = [[] for _ in range(self.p.ent_num)]
		for ent_id in ent2evt:
			for evt_id in ent2evt[ent_id]:
				ent2evt_list[ent_id].append(evt_id)
		self.ent2evt_list = ent2evt_list
		# print(ent2evt_list)

		# max_evt_num = max([len(ent2evt[key]) for key in ent2evt])

		# entity_event_index = torch.zeros(max_evt_num, self.p.ent_num, dtype=torch.long).to(self.device)
		# role_mask = torch.zeros(max_evt_num, self.p.ent_num).to(self.device)

		entity_event_index = [[] for _ in range(self.p.ent_num)]
		entity_mask = [[] for _ in range(self.p.ent_num)]


		for event_idx in self.event_edges:
			end_ents = self.event_edges[event_idx]
			for i,arg in enumerate(end_ents):
				event_index[event_idx][i] = arg[0]
				role_type[event_idx][i] = arg[1]
				role_mask[event_idx][i] = 1.0

				entity_event_index[arg[0]].append(event_idx)
				entity_mask[arg[0]].append(1.0)
		
		event_index = event_index[:, 0:self.p.entity_sample_num]
		role_type = role_type[:, 0:self.p.entity_sample_num]
		role_mask = role_mask[:, 0:self.p.entity_sample_num]
		# embed to tensor
		entity_event_index_new = entity_event_index.copy()
		entity_mask_new = entity_mask.copy()
		
		max_event_num = max([len(i) for i in entity_event_index])

		for i,l in enumerate(entity_event_index_new):
			entity_event_index[i].extend([0] * (max_event_num - len(l)))

		for i,l in enumerate(entity_mask_new):
			entity_mask[i].extend([0.0] * (max_event_num - len(l)))
		entity_event_index = torch.LongTensor(entity_event_index)[:, 0:self.p.event_sample_num].to(self.device)
		entity_mask = torch.Tensor(entity_mask)[:, 0:self.p.event_sample_num].to(self.device)

		with open("./data/" + self.p.dataset + "/temp.json", "r", encoding="utf-8") as f:
			event_edge_index = json.loads(f.read())
		self.event_edge_list = event_edge_index
		# print(event_edge_index)
		event_neigh_list = [[] for _ in range(self.p.event_num)]

		for i in range(len(event_edge_index[0])):
			event_neigh_list[event_edge_index[1][i]].append(event_edge_index[0][i])
		
		self.event_neigh_list = event_neigh_list

		event_edge_index = torch.LongTensor(event_edge_index).to(self.device)


		return event_edge_index, event_index, role_type, role_mask, entity_event_index, entity_mask

	def __init__(self, params):
		"""
		Constructor of the runner class

		Parameters
		----------
		params:         List of hyper-parameters of the model
		
		Returns
		-------
		Creates computational graph and optimizer
		
		"""
		self.p			= params
		self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p))
		# pprint(vars(self.p))

		# if self.p.gpu != '-1' and torch.cuda.is_available():
		# 	self.device = torch.device('cuda:'+self.p.gpu)
		# 	torch.cuda.set_rng_state(torch.cuda.get_rng_state())
		# 	torch.backends.cudnn.deterministic = True
		# else:
		# 	self.device = torch.device('cpu')
		
		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = int(self.p.gpu)
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.load_data()
		self.model        = self.add_model(self.p.model, self.p.score_func)
		self.optimizer    = self.add_optimizer(self.model.parameters())


	def add_model(self, model, score_func):
		"""
		Creates the computational graph

		Parameters
		----------
		model_name:     Contains the model name to be created
		
		Returns
		-------
		Creates the computational graph for model and initializes it
		
		"""
		model_name = '{}_{}'.format(model, score_func)

		if   model_name.lower()	== 'compgcn_transe': 	model = CompGCN_TransE(self.event_edge_index, self.edge_index, self.edge_type, self.event_index, self.role_type, self.role_mask, self.entity_event_index, self.entity_mask, params=self.p)
		elif model_name.lower()	== 'compgcn_distmult': 	model = CompGCN_DistMult(self.event_edge_index, self.edge_index, self.edge_type, self.event_index, self.role_type, self.role_mask, self.entity_event_index, self.entity_mask, params=self.p)
		elif model_name.lower()	== 'compgcn_conve': 	model = CompGCN_ConvE(self.event_edge_index, self.edge_index, self.edge_type, self.event_index, self.role_type, self.role_mask, self.entity_event_index, self.entity_mask, params=self.p)
		else: raise NotImplementedError
		model.to(self.device)
		return model

	def add_optimizer(self, parameters):
		"""
		Creates an optimizer for training the parameters

		Parameters
		----------
		parameters:         The parameters of the model
		
		Returns
		-------
		Returns an optimizer for learning the parameters of the model
		
		"""
		return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

	def read_batch(self, batch, split):
		# print(batch)
		"""
		Function to read a batch of data and move the tensors in batch to CPU/GPU

		Parameters
		----------
		batch: 		the batch to process
		split: (string) If split == 'train', 'valid' or 'test' split

		
		Returns
		-------
		Head, Relation, Tails, labels
		"""
		if split == 'train':
			triple, label = [ _.to(self.device) for _ in batch[0:2]]
			obj_list = batch[-1]
			return triple[:, 0], triple[:, 1], triple[:, 2], label, obj_list
		else:
			triple, label = [ _.to(self.device) for _ in batch[0:2]]
			obj_list = batch[-1]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	def save_model(self, save_path):
		"""
		Function to save a model. It saves the model parameters, best validation scores,
		best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

		Parameters
		----------
		save_path: path where the model is saved
		
		Returns
		-------
		"""
		state = {
			'state_dict'	: self.model.state_dict(),
			# 'best_val'	: self.best_val,
			# 'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			# 'args'		: vars(self.p)
		}
		torch.save(state, save_path)

	def load_model(self, load_path):
		"""
		Function to load a saved model

		Parameters
		----------
		load_path: path to the saved model
		
		Returns
		-------
		"""
		state			= torch.load(load_path)
		state_dict		= state['state_dict']
		# self.best_val		= state['best_val']
		# self.best_val_mrr	= self.best_val['mrr'] 

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])

	def evaluate(self, split, epoch):
		"""
		Function to evaluate the model on validation or test set

		Parameters
		----------
		split: (string) If split == 'valid' then evaluate on the validation set, else the test set
		epoch: (int) Current epoch count
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		if self.p.eval == "all":
			left_results  = self.predict_all(split=split, mode='tail_batch')
			right_results = self.predict_all(split=split, mode='head_batch')
		else:
			left_results  = self.predict_selected(split=split, mode='tail_batch')
			right_results = self.predict_selected(split=split, mode='head_batch')

		results       = get_combined_results(left_results, right_results)
		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
		self.logger.info(json.dumps({key:results[key] for key in ["mrr", "mr", "hits@1", "hits@3", "hits@5", "hits@10", "hits@20"]}))
		return results

	def predict_all(self, split='valid', mode='tail_batch'):
		"""
		Function to run model evaluation for a given mode

		Parameters
		----------
		split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
		mode: (string):		Can be 'head_batch' or 'tail_batch'
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		self.model.eval()

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

			total_sub, total_rel, total_obj, total_ranks = [], [], [], []

			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_batch(batch, split)

				total_sub = total_sub + sub.tolist()
				total_rel = total_rel + rel.tolist()
				total_obj = total_obj + obj.tolist()

				# sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = self.sample_subgraph(sub, objs)
				# sub, rel, obj: torch.LongTensor(batch_size)
				# label: torch.FloatTensor(batch_size, entity_num)
				pred			= self.model.predict(sub, rel)
				# print(pred.shape)
				# pred shape: (batch_size, entity_num)
				b_range			= torch.arange(pred.size()[0], device=self.device)
				# b_range: torch.LongTensor (batch_size)
				target_pred		= pred[b_range, obj]
				# target_pred 选取出来的正确位置的预测概率（注意是obj对应的唯一正确概率，而不是label对应的多个正确概率）
				pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
				# 把label对应为1位置的pred的值换成1
				pred[b_range, obj] 	= target_pred
				# print(label.shape)
				# print(torch.sum(label, 1))torch.argsort(aa, dim=1, descending=False)
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
				total_ranks += ranks.tolist()
				# 这个rank的意思是：因为label可以有多个，但是obj只有一个，这个rank代表的意思是，出去label中的其他项的干扰之后，obj的排名情况
				ranks 			= ranks.float()
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)
				# print(results)
				if step % 100 == 0:
					self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))
			
			results['ranks'] = total_ranks
			results['sub'] = total_sub
			results['rel'] = total_rel
			results['obj'] = total_obj

		return results
	
	def generate_neg_samples(self, input_list, num):
		if len(input_list) < num:
			if_enough = False
			new_list = input_list.copy()
			full = False
			while not full:
				i = random.randint(0, self.p.num_ent-1)
				if i not in new_list:
					new_list.append(i)
				if len(new_list) == num:
					full = True
			return new_list, if_enough
		else:
			if_enough = True
			return input_list[0: num].copy(), if_enough

	def predict_selected(self, split='valid', mode='tail_batch'):
		"""
		Function to run model evaluation for a given mode

		Parameters
		----------
		split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
		mode: (string):		Can be 'head_batch' or 'tail_batch'
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		# self.model.to("cpu")
		# self.model.eval()
		# self.old_device = self.model.device
		# # print(old_device)
		# self.model.device = "cpu"
		# self.model.event_conv1.device = "cpu"
		# self.model.conv1.device = "cpu"
		# self.device = "cpu"

		# self.model.event_edge_index = self.model.event_edge_index.to("cpu")
		# self.model.edge_index = self.model.edge_index.to("cpu")
		# self.model.edge_type = self.model.edge_type.to("cpu")
		# self.model.role_type = self.model.role_type.to("cpu")
		# self.model.role_mask = self.model.role_mask.to("cpu")
		# self.model.entity_event_index = self.model.entity_event_index.to("cpu")
		# self.model.entity_mask = self.model.entity_mask.to("cpu")
		# print("wryyyyyyyy")
		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

			total_sub, total_rel, total_obj, total_ranks = [], [], [], []

			for step, batch in enumerate(train_iter):
				if self.p.dataset == "kairos":
					if step > 98:
						break
				sub, rel, obj, label	= self.read_batch(batch, split)
				
				sub_list = sub.tolist()
				obj_list = obj.tolist()

				init_ent_list = []
				sub_index_list = []
				obj_index_list = []

				node_pos_dict = {}

				# rel_evt_idxs ------ final_ent_list
				for j,idx in enumerate(sub_list):
					if idx not in node_pos_dict:
						init_ent_list.append(idx)
						node_pos_dict.update({idx: len(init_ent_list) - 1})
						sub_index_list.append(len(init_ent_list) - 1)
					else:
						sub_index_list.append(node_pos_dict[idx])

				# print(sub_index_list)
				# print(objs)
				for j,idx in enumerate(obj_list):
					if idx not in node_pos_dict:
						init_ent_list.append(idx)
						node_pos_dict.update({idx: len(init_ent_list) - 1})
						obj_index_list.append(len(init_ent_list) - 1)
					else:
						obj_index_list.append(node_pos_dict[idx])

				final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = self.sample_subgraph(init_ent_list)

				# print(len(final_ent_list))
				# print(len(ent_neighbors))
				# print(len(rel_evt_idxs))
				# print(len(rel_evt_idxs))

				selected_list, if_enough = self.generate_neg_samples(final_ent_list, 500)

				if not if_enough:
					final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = self.sample_subgraph(selected_list)

				new_label = label[:, selected_list]
				# no matter how, we selected top 500
				total_sub = total_sub + sub.tolist()
				total_rel = total_rel + rel.tolist()
				total_obj = total_obj + obj.tolist()

				# sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = self.sample_subgraph(sub, objs)
				# sub, rel, obj: torch.LongTensor(batch_size)
				# label: torch.FloatTensor(batch_size, entity_num)
				pred			= self.model.forward(sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type)

				pred = pred[:, 0:len(selected_list)]

				# print(pred.shape)
				# pred shape: (batch_size, entity_num)
				b_range			= torch.arange(pred.size()[0], device=self.device)
				# b_range			= torch.arange(pred.size()[0], device="cpu")
				# b_range: torch.LongTensor (batch_size)
				target_pred		= pred[b_range, obj_index_list]
				# target_pred 选取出来的正确位置的预测概率（注意是obj对应的唯一正确概率，而不是label对应的多个正确概率）
				pred 			= torch.where(new_label.byte(), -torch.ones_like(pred) * 10000000, pred)
				# 把label对应为1位置的pred的值换成1
				pred[b_range, obj_index_list] 	= target_pred
				# print(label.shape)
				# print(torch.sum(label, 1))torch.argsort(aa, dim=1, descending=False)
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj_index_list]
				total_ranks += ranks.tolist()
				# 这个rank的意思是：因为label可以有多个，但是obj只有一个，这个rank代表的意思是，出去label中的其他项的干扰之后，obj的排名情况
				ranks 			= ranks.float()
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)
				
				results['hits@{}'.format(20)] = torch.numel(ranks[ranks <= (20)]) + results.get('hits@{}'.format(20), 0.0)

				# print(results)
				if step % 100 == 0:
					self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))
			
			results['ranks'] = total_ranks
			results['sub'] = total_sub
			results['rel'] = total_rel
			results['obj'] = total_obj

			# print(results)


		# self.model.device = self.old_device
		# self.device = self.old_device
		# self.model.event_conv1.device = self.old_device
		# self.model.conv1.device = self.old_device
		# self.model.to(self.old_device)

		# self.model.event_edge_index = self.model.event_edge_index.to(self.old_device)
		# self.model.edge_index = self.model.edge_index.to(self.old_device)
		# self.model.edge_type = self.model.edge_type.to(self.old_device)
		# self.model.role_type = self.model.role_type.to(self.old_device)
		# self.model.role_mask = self.model.role_mask.to(self.old_device)
		# self.model.entity_event_index = self.model.entity_event_index.to(self.old_device)
		# self.model.entity_mask = self.model.entity_mask.to(self.old_device)

		return results

	def sample_subgraph(self, init_ent_list):
		# print(1)
		rel_evt_idxs = []
		for ent_idx in init_ent_list:
			rel_evt_idxs = rel_evt_idxs + self.ent2evt_list[ent_idx]
		rel_evt_idxs = list(set(rel_evt_idxs))
		if self.p.dataset == "kairos":
			rel_evt_idxs = rel_evt_idxs[0:5000]
		event_ent_idxs = []

		for evt_idx in rel_evt_idxs:
			event_ent_idxs += self.evt2ent_list[evt_idx]

		final_ent_list = init_ent_list.copy()
		if self.p.dataset == "kairos":
			pass
		else:
			for idx in event_ent_idxs:
				if idx not in final_ent_list:
					final_ent_list.append(idx)

		ent_neighbors, evt_neighbors = final_ent_list.copy(), rel_evt_idxs.copy()

		if self.p.dataset == "kairos":
			for ent_idx in final_ent_list:
				idxs = self.edge_list[ent_idx]
				for idx in idxs:
					if idx not in ent_neighbors:
						ent_neighbors.append(idx)
		
			ent_neighbors = ent_neighbors[0:10000]

		else:
			# ent_neighbors, evt_neighbors
			for ent_idx in final_ent_list:
				idxs = self.edge_list[ent_idx]
				for idx in idxs:
					if idx not in ent_neighbors:
						ent_neighbors.append(idx)
			
			for evt_idx in rel_evt_idxs:
				idxs = self.event_neigh_list[evt_idx]
				for idx in idxs:
					if idx not in evt_neighbors:
						evt_neighbors.append(idx)
			
			for ent_id in ent_neighbors:
				idxs = self.ent2evt_list[ent_id]
				for idx in idxs:
					if idx not in evt_neighbors:
						evt_neighbors.append(idx)
			
			for evt_id in evt_neighbors:
				idxs = self.evt2ent_list[evt_id]
				for idx in idxs:
					if idx not in ent_neighbors:
						ent_neighbors.append(idx)


		# assert (rel_evt_idxs == evt_neighbors[0:len(rel_evt_idxs)])
		# assert (final_ent_list == ent_neighbors[0:len(final_ent_list)])
		# print(4)
		
		evt_total_to_selected = [-1 for _ in range(self.p.event_num)]
		for j,evt_id in enumerate(evt_neighbors):
			evt_total_to_selected[evt_id] = j
		# print(5)
		
		entity_mask_list = self.entity_mask[ent_neighbors].tolist()
		entity_event_index_list = self.entity_event_index[ent_neighbors].tolist()

		new_entity_event_index_list = []
		for i in range(len(entity_event_index_list)):
			events_i = []
			for j in range(len(entity_event_index_list[0])):
				evt_mask = entity_mask_list[i][j]
				evt_idx = entity_event_index_list[i][j]
				if evt_mask != 0.0:
					if evt_total_to_selected[evt_idx] == -1:
						events_i.append(0)
						entity_mask_list[i][j] = 0.0
					else:
						events_i.append(evt_total_to_selected[evt_idx])
				else:
					events_i.append(0)

			new_entity_event_index_list.append(events_i)
		# print(6)
		
		new_entity_event_index = torch.LongTensor(new_entity_event_index_list).to(self.device)
		new_entity_mask = torch.FloatTensor(entity_mask_list).to(self.device)

		if self.p.dataset == "kairos":

			new_event_list = [[], []]
			for i in range(len(self.event_edge_list[0])):
				start = self.event_edge_list[0][i]
				end = self.event_edge_list[1][i]
				if end in rel_evt_idxs and start in rel_evt_idxs:
					new_event_list[0].append(evt_total_to_selected[start])
					new_event_list[1].append(evt_total_to_selected[end])
		else:
			new_event_list = [[], []]
			for i in range(len(self.event_edge_list[0])):
				start = self.event_edge_list[0][i]
				end = self.event_edge_list[1][i]
				if end in rel_evt_idxs:
					new_event_list[0].append(evt_total_to_selected[start])
					new_event_list[1].append(evt_total_to_selected[end])
		# print(7)

		new_event_index = torch.LongTensor(new_event_list).to(self.device)

		# calculate the entities
		ent_total_to_selected = [-1 for _ in range(self.p.ent_num)]
		for j,ent_id in enumerate(ent_neighbors):
			ent_total_to_selected[ent_id] = j
		# print(8)
		
		# calculate the new entity edges
		new_entity_list = [[], []]
		new_entity_type = []

		edge_index_list = self.edge_index.tolist()
		edge_type_list = self.edge_type.tolist()

		if self.p.dataset == "kairos":

			for i in range(len(edge_index_list[0])):
				start = edge_index_list[0][i]
				end = edge_index_list[1][i]
				if end in final_ent_list and start in ent_neighbors:
					new_entity_list[0].append(ent_total_to_selected[start])
					new_entity_list[1].append(ent_total_to_selected[end])
					new_entity_type.append(edge_type_list[i])
		else:

			for i in range(len(edge_index_list[0])):
				start = edge_index_list[0][i]
				end = edge_index_list[1][i]
				if end in final_ent_list:
					new_entity_list[0].append(ent_total_to_selected[start])
					new_entity_list[1].append(ent_total_to_selected[end])
					new_entity_type.append(edge_type_list[i])
		# print(9)
		
		new_entity_list = torch.LongTensor(new_entity_list).to(self.device)
		new_entity_type = torch.LongTensor(new_entity_type).to(self.device)

		return final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type


	def run_epoch(self, epoch, val_mrr = 0):
		# print(self.evt2ent_list)
		numnum = sum([len(i) for i in self.evt2ent_list])
		print("args", numnum)
		print("args_type", len(self.role2id))
		print("rel_type", len(self.rel2id) // 2)
		print("ent_num", self.p.ent_num)


		"""
		Function to run one epoch of training

		Parameters
		----------
		epoch: current epoch count
		
		Returns
		-------
		loss: The loss value after the completion of one epoch
		"""
		self.model.train()
		losses = []
		train_iter = iter(self.data_iter['train'])
		# we have: self.ent2evt_list, self.evt2ent_list, self.event_neigh_list, self.edge_list
		for step, batch in enumerate(train_iter):
			if self.p.dataset == "kairos":
				if step > 98:
					print("kairos break")
					break
			# print("step: ", step)
			self.optimizer.zero_grad()
			sub, rel, obj, label, objs = self.read_batch(batch, 'train')

			sub_list = sub.tolist()
			init_ent_list = []
			sub_index_list = []

			node_pos_dict = {}

			# rel_evt_idxs ------ final_ent_list
			for j,idx in enumerate(sub_list):
				if idx not in node_pos_dict:
					init_ent_list.append(idx)
					node_pos_dict.update({idx: len(init_ent_list) - 1})
					sub_index_list.append(len(init_ent_list) - 1)
				else:
					sub_index_list.append(node_pos_dict[idx])
			# print(sub_index_list)
			# print(objs)
			for idx in objs:
				if idx not in init_ent_list:
					init_ent_list.append(idx)

			if self.p.train_all == '1':
				# print("yes")
				final_ent_list = [ _ for _ in range(self.p.ent_num)]
				ent_neighbors = [ _ for _ in range(self.p.ent_num)]
				rel_evt_idxs = [ _ for _ in range(self.p.event_num)]
				evt_neighbors = [ _ for _ in range(self.p.event_num)]

				new_event_index = self.event_edge_index
				new_entity_event_index = self.entity_event_index
				new_entity_mask = self.entity_mask
				new_entity_list = self.edge_index
				new_entity_type = self.edge_type
				sub_index_list = sub.tolist()

			else:
				
				final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = self.sample_subgraph(init_ent_list)
				# print(len(final_ent_list))
				# print(len(ent_neighbors))
				# print(len(rel_evt_idxs))
				# print(len(rel_evt_idxs))

			new_label = label[:, final_ent_list]
			pred	= self.model.forward(sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type)
			# print(pred.shape)
			loss	= self.model.loss(pred, new_label)

			loss.backward()
			self.optimizer.step()
			losses.append(loss.item())

			if step % 100 == 0:
				self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

		loss = np.mean(losses)
		self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
		return loss


	def fit(self):
		"""
		Function to run training and evaluation of model

		Parameters
		-------
		Returns
		-------
		"""

		self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
		save_path = os.path.join('./checkpoints', (self.p.dataset + '_' +self.p.name))
		entity_path = os.path.join('./checkpoints', (self.p.dataset + '_' +self.p.name) +"_entity_typing")
		relation_path = os.path.join('./checkpoints', (self.p.dataset + '_' +self.p.name) +"_relation_typing")

		results_path = os.path.join('./results', (self.p.dataset + '_' +self.p.name) + ".json")

		entity_analysis_path = os.path.join('./analysis/entity', (self.p.dataset + '_' +self.p.name) + ".json")
		relation_analysis_path = os.path.join('./analysis/relation', (self.p.dataset + '_' +self.p.name) + ".json")
		kg_analysis_path = os.path.join('./analysis/kg', (self.p.dataset + '_' +self.p.name) + ".json")

		if self.p.restore:
			self.load_model(save_path)
			self.logger.info('Successfully Loaded previous model')

		kill_cnt = 0
		for epoch in range(self.p.max_epochs):
			train_loss  = self.run_epoch(epoch, val_mrr)
			# train_loss  = 0.0
			# print("validation")
			val_results = self.evaluate('valid', epoch)

			if val_results['mrr'] > self.best_val_mrr:
				self.best_val	   = val_results
				self.best_val_mrr  = val_results['mrr']
				self.best_epoch	   = epoch
				self.save_model(save_path)
				kill_cnt = 0
			else:
				kill_cnt += 1
				if kill_cnt % 10 == 0 and self.p.gamma > 5:
					self.p.gamma -= 5 
					self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
				if kill_cnt > 25: 
					self.logger.info("Early Stopping!!")
					break

			self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))

		self.logger.info('Loading best model, Evaluating on Test data')
		self.load_model(save_path)

		# first store val results
		epoch = 0
		val_results = self.evaluate('valid', epoch)
		val_subs = val_results["sub"]
		val_rels = val_results["rel"]
		val_objs = val_results["obj"]
		with open(kg_analysis_path, "w", encoding="utf-8") as f:
			for i in range(len(val_results["left_ranks"])):
				start_events = self.ent2evt.get(val_subs[i], ["None"])
				start_related_events = []
				for evt in start_events:
					if evt != "None":
						start_related_events.append(self.id2evt[evt])

				end_events = self.ent2evt.get(val_objs[i], ["None"])
				end_related_events = []
				for evt in end_events:
					if evt != "None":
						end_related_events.append(self.id2evt[evt])

				item_i = {"start_entity": self.id2ent[val_subs[i]], "relation": self.id2rel[val_rels[i]], "end_entity": self.id2ent[val_objs[i]], "start_related_events": start_related_events, "end_related_events": end_related_events, "left_rank": val_results["left_ranks"][i], "right_rank": val_results["right_ranks"][i]}
				f.write(json.dumps(item_i, indent=4, separators=(',',':')) + '\n')

		test_results = self.evaluate('test', epoch)
		test_results.update({"epoch": self.best_epoch})

		# experiments for entity typing

		max_eval_loss = np.inf
		max_eval_accu = 0.0

		num_not_decrease = 0

		# load entity typing datasets

		train_ent_ds = TensorDataset(self.train_ent_ids.unsqueeze(-1), self.train_ent_labels.unsqueeze(-1))
		train_ent_dl = DataLoader(train_ent_ds, batch_size=self.p.rel_batch_size, shuffle=True)

		valid_ent_ds = TensorDataset(self.valid_ent_ids.unsqueeze(-1), self.valid_ent_labels.unsqueeze(-1))
		valid_ent_dl = DataLoader(valid_ent_ds, batch_size=self.p.rel_batch_size, shuffle=False)

		test_ent_ds = TensorDataset(self.test_ent_ids.unsqueeze(-1), self.test_ent_labels.unsqueeze(-1))
		test_ent_dl = DataLoader(test_ent_ds, batch_size=self.p.rel_batch_size, shuffle=False)


		for epoch in range(self.p.max_epochs):
		# for epoch in range(5):
			self.model.train()
			train_batch_loss = 0.0
			train_batch_num = 0
			step = 0
			for ent_idxs, label_idxs in train_ent_dl:
				step += 1
				if self.p.dataset == "kairos":
					if step > 20:
						print("kairos break")
						break
				# print(1)
				# print(ent_idxs.tolist())
				batch_ent_list = ent_idxs.squeeze(1).tolist()
				final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = self.sample_subgraph(batch_ent_list)
				# print(len(batch_ent_list))

				# print(new_entity_type.device)
				train_loss = self.model.forward_for_entity_typing(ent_idxs, label_idxs, batch_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type, predict=False)

				self.optimizer.zero_grad()
				train_loss.backward()
				self.optimizer.step()

				train_batch_loss += train_loss.item()
				train_batch_num += 1
			
			train_batch_loss = train_batch_loss / train_batch_num

			correct_num = 0.0
			total_num = 0.0

			self.model.eval()
			step = 0
			for ent_idxs, label_idxs in valid_ent_dl:
				step += 1
				if self.p.dataset == "kairos":
					if step > 20:
						print("kairos break")
						break
				val_ent_list = ent_idxs.squeeze(1).tolist()
				final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = self.sample_subgraph(val_ent_list)

				eval_loss, eval_pred = self.model.forward_for_entity_typing(ent_idxs, label_idxs, val_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type, predict=True)

				correct_num += torch.sum((eval_pred == label_idxs.squeeze(1)).float()).item()
				total_num += eval_pred.shape[0]
			
			eval_accu = correct_num / total_num
			
			self.logger.info("Epoch " + str(epoch + 1) + " training_loss: " + str(train_loss.item()) + ", eval_loss: " + str(eval_loss.item()) + ", eval_accuracy: " + str(eval_accu))

			if eval_accu > max_eval_accu:
				max_eval_accu = eval_accu
				num_not_decrease = 0
				self.save_model(entity_path)
			else:
				num_not_decrease += 1
			
			if num_not_decrease >= 20:
				print("Early Stopping!")
				break

		self.load_model(entity_path)
		self.model.eval()
		
		correct_num = 0.0
		total_num = 0.0

		# run validation again
		entity_ids = []
		entity_type_ids = []
		entity_pred_ids = []

		step = 0
		for ent_idxs, label_idxs in valid_ent_dl:
			step += 1
			if self.p.dataset == "kairos":
				if step > 20:
					print("kairos break")
					break
			val_ent_list = ent_idxs.squeeze(1).tolist()
			final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = self.sample_subgraph(val_ent_list)

			entity_ids += ent_idxs.squeeze(1).tolist()
			entity_type_ids += label_idxs.squeeze(1).tolist()

			eval_loss, eval_pred = self.model.forward_for_entity_typing(ent_idxs, label_idxs, val_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type, predict=True)

			entity_pred_ids += eval_pred.tolist()
			correct_num += torch.sum((eval_pred == label_idxs.squeeze(1)).float()).item()
			total_num += eval_pred.shape[0]
		
		with open(entity_analysis_path, "w", encoding="utf-8") as f:
			for i in range(len(entity_ids)):

				end_events = self.ent2evt.get(entity_ids[i], ["None"])
				end_related_events = []
				for evt in end_events:
					if evt != "None":
						end_related_events.append(self.id2evt[evt])

				item_i = {"entity_id": self.id2ent[entity_ids[i]], "correct_type": self.id_to_ent_type[entity_type_ids[i]], "predict_type": self.id_to_ent_type[entity_pred_ids[i]], "related_events": end_related_events}
				f.write(json.dumps(item_i, indent=4, separators=(',',':')) + '\n')

		# testing
		for ent_idxs, label_idxs in test_ent_dl:
			val_ent_list = ent_idxs.squeeze(1).tolist()
			final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = self.sample_subgraph(val_ent_list)

			eval_loss, eval_pred = self.model.forward_for_entity_typing(ent_idxs, label_idxs, val_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type, predict=True)

			correct_num += torch.sum((eval_pred == label_idxs.squeeze(1)).float()).item()
			total_num += eval_pred.shape[0]
		

		ent_accuracy = correct_num / total_num
		print("Final testing accuracy: " + str(ent_accuracy))
		self.logger.info("Final testing accuracy: " + str(ent_accuracy))

		# experiments for relation classification
		self.logger.info("Start training for relation typing")

		# load model from save_path
		self.load_model(save_path)

		train_rel_ds = TensorDataset(self.train_rel_start, self.train_rel_end, self.train_rel_labels)
		train_rel_dl = DataLoader(train_rel_ds, batch_size=self.p.rel_batch_size, shuffle=True)

		valid_rel_ds = TensorDataset(self.valid_rel_start, self.valid_rel_end, self.valid_rel_labels)
		valid_rel_dl = DataLoader(valid_rel_ds, batch_size=self.p.rel_batch_size, shuffle=False)

		test_rel_ds = TensorDataset(self.test_rel_start, self.test_rel_end, self.test_rel_labels)
		test_rel_dl = DataLoader(test_rel_ds, batch_size=self.p.rel_batch_size, shuffle=False)

		max_eval_accu = 0.0
		num_not_decrease = 0

		for epoch in range(self.p.max_epochs):
		# for epoch in range(5):

			self.model.train()
			# training batches
			train_batch_loss = 0.0
			train_batch_num = 0
			step = 0
			for start_idxs, end_idxs, label_idxs in train_rel_dl:
				# merge the lists\
				# print(1)
				step += 1
				if self.p.dataset == "kairos":
					if step > 20:
						print("kairos break")
						break
				node_idxs_list = []
				start_index_list, end_index_list = [], []

				node_pos_dict = {}

				batch_start_list = start_idxs.squeeze(1).tolist()
				batch_end_list = end_idxs.squeeze(1).tolist()

				for start in batch_start_list:
					if start not in node_pos_dict:
						node_idxs_list.append(start)
						node_pos_dict.update({start: len(node_idxs_list)-1})
						start_index_list.append(len(node_idxs_list)-1)
					else:
						start_index_list.append(node_pos_dict[start])
				
				for end in batch_end_list:
					if end not in node_pos_dict:
						node_idxs_list.append(end)
						node_pos_dict.update({end: len(node_idxs_list)-1})
						end_index_list.append(len(node_idxs_list)-1)
					else:
						end_index_list.append(node_pos_dict[end])
				
				# print(end_index_list)

				final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = self.sample_subgraph(node_idxs_list)

				
				# assert (batch_start_list[10] == final_ent_list[start_index_list[10]])

				train_loss = self.model.forward_for_relation_typing(start_idxs, end_idxs, label_idxs, start_index_list, end_index_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type, predict=False)

				self.optimizer.zero_grad()
				train_loss.backward()
				self.optimizer.step()

				train_batch_loss += train_loss.item()
				train_batch_num += 1
			
			train_batch_loss = train_batch_loss / train_batch_num

			# validation
			correct_num = 0.0
			total_num = 0.0

			self.model.eval()
			step = 0
			for start_idxs, end_idxs, label_idxs in valid_rel_dl:
				step += 1
				if self.p.dataset == "kairos":
					if step > 20:
						print("kairos break")
						break
				node_idxs_list = []
				start_index_list, end_index_list = [], []

				node_pos_dict = {}

				batch_start_list = start_idxs.squeeze(1).tolist()
				batch_end_list = end_idxs.squeeze(1).tolist()

				for start in batch_start_list:
					if start not in node_pos_dict:
						node_idxs_list.append(start)
						node_pos_dict.update({start: len(node_idxs_list)-1})
						start_index_list.append(len(node_idxs_list)-1)
					else:
						start_index_list.append(node_pos_dict[start])
				
				for end in batch_end_list:
					if end not in node_pos_dict:
						node_idxs_list.append(end)
						node_pos_dict.update({end: len(node_idxs_list)-1})
						end_index_list.append(len(node_idxs_list)-1)
					else:
						end_index_list.append(node_pos_dict[end])

				# for i in range(len(batch_start_list)):
				# 	print(batch_start_list[i])
				# 	print(final_ent_list[start_index_list[i]])

				# 	assert (batch_start_list[i] == final_ent_list[start_index_list[i]])
				final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = self.sample_subgraph(node_idxs_list)
				
				eval_loss, eval_pred = self.model.forward_for_relation_typing(start_idxs, end_idxs, label_idxs, start_index_list, end_index_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type, predict=True)

				

				correct_num += torch.sum((eval_pred == label_idxs.squeeze(1)).float()).item()
				total_num += eval_pred.shape[0]
			
			eval_accu = correct_num / total_num

			self.logger.info("Epoch " + str(epoch + 1) + " training_loss: " + str(train_loss.item()) + ", eval_accuracy: " + str(eval_accu))

			if eval_accu > max_eval_accu:
				max_eval_accu = eval_accu
				num_not_decrease = 0
				self.save_model(relation_path)
			else:
				num_not_decrease += 1
			
			if num_not_decrease >= 20:
				print("Early Stopping!")
				break

		self.load_model(relation_path)
		self.model.eval()
		
		correct_num = 0.0
		total_num = 0.0
		
		start_ids, end_ids = [], []
		rel_type_ids = []
		rel_pred_ids = []

		step = 0
		for start_idxs, end_idxs, label_idxs in valid_rel_dl:
			step += 1
			if self.p.dataset == "kairos":
				if step > 20:
					print("kairos break")
					break
			node_idxs_list = []
			start_index_list, end_index_list = [], []

			node_pos_dict = {}

			batch_start_list = start_idxs.squeeze(1).tolist()
			batch_end_list = end_idxs.squeeze(1).tolist()

			for start in batch_start_list:
				if start not in node_pos_dict:
					node_idxs_list.append(start)
					node_pos_dict.update({start: len(node_idxs_list)-1})
					start_index_list.append(len(node_idxs_list)-1)
				else:
					start_index_list.append(node_pos_dict[start])
			
			for end in batch_end_list:
				if end not in node_pos_dict:
					node_idxs_list.append(end)
					node_pos_dict.update({end: len(node_idxs_list)-1})
					end_index_list.append(len(node_idxs_list)-1)
				else:
					end_index_list.append(node_pos_dict[end])

			start_ids += start_idxs.squeeze(1).tolist()
			end_ids += end_idxs.squeeze(1).tolist()
			rel_type_ids += label_idxs.squeeze(1).tolist()

			final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = self.sample_subgraph(node_idxs_list)

			eval_loss, eval_pred = self.model.forward_for_relation_typing(start_idxs, end_idxs, label_idxs, start_index_list, end_index_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type, predict=True)

			rel_pred_ids += eval_pred.tolist()
		
		with open(relation_analysis_path, "w", encoding="utf-8") as f:
			for i in range(len(start_ids)):
				start_events = self.ent2evt.get(start_ids[i], ["None"])
				start_related_events = []
				for evt in start_events:
					if evt != "None":
						start_related_events.append(self.id2evt[evt])

				end_events = self.ent2evt.get(end_ids[i], ["None"])
				end_related_events = []
				for evt in end_events:
					if evt != "None":
						end_related_events.append(self.id2evt[evt])

				item_i = {"start_entity_id": self.id2ent[start_ids[i]], "end_entity_id": self.id2ent[end_ids[i]], "correct_type": self.id2rel[rel_type_ids[i]], "predict_type": self.id2rel[rel_pred_ids[i]], "start_related_events": start_related_events, "end_related_events": end_related_events}
				f.write(json.dumps(item_i, indent=4, separators=(',',':')) + '\n')


		# testing
		for start_idxs, end_idxs, label_idxs in test_rel_dl:

			node_idxs_list = []
			start_index_list, end_index_list = [], []

			node_pos_dict = {}

			batch_start_list = start_idxs.squeeze(1).tolist()
			batch_end_list = end_idxs.squeeze(1).tolist()

			for start in batch_start_list:
				if start not in node_pos_dict:
					node_idxs_list.append(start)
					node_pos_dict.update({start: len(node_idxs_list)-1})
					start_index_list.append(len(node_idxs_list)-1)
				else:
					start_index_list.append(node_pos_dict[start])
			
			for end in batch_end_list:
				if end not in node_pos_dict:
					node_idxs_list.append(end)
					node_pos_dict.update({end: len(node_idxs_list)-1})
					end_index_list.append(len(node_idxs_list)-1)
				else:
					end_index_list.append(node_pos_dict[end])

			start_ids += start_idxs.squeeze(1).tolist()
			end_ids += end_idxs.squeeze(1).tolist()
			rel_type_ids += label_idxs.squeeze(1).tolist()

			final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type = self.sample_subgraph(node_idxs_list)

			eval_loss, eval_pred = self.model.forward_for_relation_typing(start_idxs, end_idxs, label_idxs, start_index_list, end_index_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type, predict=True)

			correct_num += torch.sum((eval_pred == label_idxs.squeeze(1)).float()).item()
			total_num += eval_pred.shape[0]
		

		rel_accuracy = correct_num / total_num
		print("Final testing accuracy: " + str(rel_accuracy))
		self.logger.info("Final testing accuracy: " + str(rel_accuracy))

		with open(results_path, "w", encoding="utf-8") as f:
			f.write(json.dumps(test_results, indent=4, separators=(',',':'))+'\n') 
			f.write(json.dumps({"entity accuracy": str(ent_accuracy)[0:7]})+'\n') 
			f.write(json.dumps({"relation accuracy": str(rel_accuracy)[0:7]})+'\n') 




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('-name',		default='testrun',					help='Set run name for saving/restoring models')
	parser.add_argument('-data',		dest='dataset',         default='FB15k-237',            help='Dataset to use, default: FB15k-237')
	parser.add_argument('-model',		dest='model',		default='compgcn',		help='Model Name')
	parser.add_argument('-score_func',	dest='score_func',	default='conve',		help='Score Function for Link prediction')
	parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')

	parser.add_argument('-batch',           dest='batch_size',      default=64,    type=int,       help='Batch size')
	parser.add_argument('-rel_batch',           dest='rel_batch_size',      default=2560,    type=int,       help='Relation Classification Batch size')
	parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin')
	parser.add_argument('-gamma1',		type=float,             default=0.1,			help='Event-event propagation hyper-param')
	parser.add_argument('-gpu',		type=str,               default='-1',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
	parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=500,  	help='Number of epochs')
	parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
	parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
	parser.add_argument('-lbl_smooth',      dest='lbl_smooth',	type=float,     default=0.1,	help='Label Smoothing')
	parser.add_argument('-num_workers',	type=int,               default=10,                     help='Number of processes to construct batches')
	parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,     	help='Seed for randomization')

	parser.add_argument('-restore',         dest='restore',         action='store_true',            help='Restore from the previously saved model')
	parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')

	parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
	parser.add_argument('-init_dim',	dest='init_dim',	default=768,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=768,   	type=int, 	help='Number of hidden units in GCN')
	parser.add_argument('-role_type_dim',	  	dest='role_type_dim', 	default=200,   	type=int, 	help='Role type embedding dimension size')
	parser.add_argument('-embed_dim',	dest='embed_dim', 	default=None,   type=int, 	help='Embedding dimension to give as input to score function')
	parser.add_argument('-entity_type_num',	dest='entity_type_num', 	default=25,   type=int, 	help='Number of entity types.')
	parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=1,   	type=int, 	help='Number of GCN Layers to use')
	parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.1,  	type=float,	help='Dropout to use in GCN Layer')
	parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0.3,  	type=float,	help='Dropout after GCN')

	# ConvE specific hyperparameters
	parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
	parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
	parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   	type=int, 	help='ConvE: k_h')
	parser.add_argument('-use_event',	  	dest='use_event', 		default=1,   	type=int, 	help='Whether to use event')
	parser.add_argument('-use_temporal',	  	dest='use_temporal', 		default=0,   	type=int, 	help='Whether to use use_temporal')
	parser.add_argument('-num_filt',  	dest='num_filt', 	default=200,   	type=int, 	help='ConvE: Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=7,   	type=int, 	help='ConvE: Kernel size to use')
	parser.add_argument('-alpha', dest='alpha', default=0.01, type=float, help='event_proportion')

	parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
	parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')
	parser.add_argument('-event_sample_num',          dest='event_sample_num',      default=10,            help='Max Event Sample Num')
	parser.add_argument('-entity_sample_num',          dest='entity_sample_num',      default=10,            help='Max Event Sample Num')
	parser.add_argument('-eval',          dest='eval',      default="selected",            help='evaluate on gpu or cpu.')
	parser.add_argument('-train_all',          dest='train_all',      default=1,            help='whether to train all in one shot')
	args = parser.parse_args()

	# if not args.restore: args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
	# set_gpu(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)

	model = Runner(args)
	model.fit()
