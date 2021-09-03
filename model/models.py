from helper import *
from model.compgcn_conv import CompGCNConv, EventConv
from model.compgcn_conv_basis import CompGCNConvBasis
import sys

class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)
		
class CompGCNBase(BaseModel):
	def __init__(self, event_edge_index, edge_index, edge_type, event_index, role_type, role_mask, entity_event_index, entity_mask, num_rel, params=None):
		super(CompGCNBase, self).__init__(params)
		self.event_edge_index = event_edge_index

		self.edge_index		= edge_index
		self.edge_type		= edge_type

		self.event_index = event_index
		self.role_type = role_type
		self.role_mask = role_mask
		self.entity_event_index = entity_event_index
		self.entity_mask = entity_mask

		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim

		if params.gpu != "-1":
			self.device		= int(params.gpu)
		else:
			self.device		= torch.device("cpu")

		# entity_typing_parameter
		self.linear_1 = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim)
		self.linear_2 = torch.nn.Linear(self.p.embed_dim, self.p.entity_type_num)

		self.rel_linear_1 = torch.nn.Linear(2 * self.p.embed_dim, self.p.embed_dim)
		self.rel_linear_2 = torch.nn.Linear(self.p.embed_dim, self.p.num_rel)

		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)

		ent_init_embed = torch.Tensor(np.load(params.entity_embed_dir))
		self.init_embed = torch.nn.Parameter(ent_init_embed)

		# self.init_embed = get_param_device(self.p.num_ent, self.p.init_dim, self.device)
		
		# if params.gpu != "-1":
		# 	self.device		= torch.device("cuda:"+params.gpu)
		# else:
		# 	self.device		= torch.device("cpu")

		if self.p.num_bases > 0:
			self.init_rel  = get_param((self.p.num_bases,   self.p.init_dim))
		else:
			if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
			else: 					self.init_rel = get_param((num_rel*2, self.p.init_dim))

		if self.p.num_bases > 0:
			self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
		else:
			self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
			# num_rel： 没有添加reverse边的数量
			# 两层GCN layer: init_dim --> gcn_dim --> embed_dim

		self.event_conv1 = EventConv(self.p.init_dim, self.p.gcn_dim, self.p.role_type_dim, self.p.role_num, act=self.act, params=self.p)
		self.event_conv2 = EventConv(self.p.gcn_dim, self.p.embed_dim, self.p.role_type_dim, self.p.role_num, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

		# self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

		self.load_events(params)

	def load_events(self, param):
		event_embed_dir = "./data/" + param.dataset + "/event_embed.npy"
		event_type_dir = "./data/" + param.dataset + "/event_types.json"
		event_ids_dir = "./data/" + param.dataset + "/event_ids.json"

		with open(event_type_dir, "r", encoding="utf-8") as f:
			self.event_types = json.loads(f.readline())
		
		with open(event_ids_dir, "r", encoding="utf-8") as f:
			self.evt2id = json.loads(f.readline())
		
		self.event_type_idxs = {}

		event_type_num = 0
		for event_id in self.event_types:
			if self.event_types[event_id] not in self.event_type_idxs:
				self.event_type_idxs.update({self.event_types[event_id]: event_type_num})
				event_type_num += 1
		
		event_type_num = len(self.event_type_idxs)
		# self.event_type_embed = get_param_device(event_type_num, param.role_type_dim, self.device)
		self.event_type_embed = torch.randn(event_type_num, param.role_type_dim).to(self.device)

		self.event2type = {}

		for event_id in self.evt2id:
			# print(event_id)
			evt_idx = self.evt2id[event_id]
			# print(evt_idx)
			self.event2type.update({evt_idx: self.event_type_idxs[self.event_types[event_id]]})
		# print(self.event_types)
		
		evt_init_embed = torch.Tensor(np.load(event_embed_dir))
		self.event_embed = torch.nn.Parameter(evt_init_embed)

		# self.event_embed = get_param_device(len(self.evt2id), self.p.init_dim, self.device)

		# load event type_idx
		event_type_idxs = [0 for _ in range(len(self.event2type))]
		for idx in self.event2type:
			event_type_idxs[idx] = self.event2type[idx]
		# event_type_idxs = torch.LongTensor(event_type_idxs).to(self.device)
		self.evt_type_emb = torch.nn.Parameter(self.event_type_embed[event_type_idxs])
		# self.evt_type_emb = self.event_type_embed[event_type_idxs]

		self.role_type_embed  = get_param_device(param.role_num, param.role_type_dim, self.device)


		# we have: 
		# self.evt_embed, self.evt2id, self.event_types

		'''
			event information for implementing forward_base:
				1) self.event_index: (max_role_num, event_num): containing event idxs and entity idxs
				2) self.role_type: (max_role_num, event_num): containing role type idxs between each event and entity pair
				3) self.role_mask: (max_role_num, event_num): masked values
				4) self.role_type_embed: (role_type_num, role_type_dim): containing role_type_embeddings

				5) self.event_embed: (event_num, init_dim): containing trigger embeddings for each event
				6) self.event2type: dict: {event_idx: event_type_idx}
				7) self.event_type_embed: (event_type_num, init_dim): containing event_type embeddings for each type
		'''

		# print(self.evt2id)

	def forward_base(self, sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type):
		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		if self.p.use_event:
			x = self.event_conv1(new_event_index, self.init_embed, self.event_index, self.role_type, self.role_mask, new_entity_event_index, new_entity_mask, self.role_type_embed, self.event_embed, self.evt_type_emb, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors)
		else:
			x = self.init_embed[ent_neighbors]

		x, r	= self.conv1(x, new_entity_list, new_entity_type, rel_embed=r)
		x	= self.hidden_drop(x)

		return x[0:len(final_ent_list)], r
	
	def predict_kg_base(self, sub, rel):	
		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)

		final_ent_list = [ _ for _ in range(self.p.ent_num)]
		ent_neighbors = [ _ for _ in range(self.p.ent_num)]
		rel_evt_idxs = [ _ for _ in range(self.p.event_num)]
		evt_neighbors = [ _ for _ in range(self.p.event_num)]

		if self.p.use_event:
			x = self.event_conv1(self.event_edge_index, self.init_embed, self.event_index, self.role_type, self.role_mask, self.entity_event_index, self.entity_mask, self.role_type_embed, self.event_embed, self.evt_type_emb, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors)
		else:
			x = self.init_embed
		# relation convolution layer 1
		x, r	= self.conv1(x, self.edge_index, self.edge_type, rel_embed=r)
		x	= self.hidden_drop(x)
		return x, r

	def forward_for_kg_completion(self, sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type):
		# print(rel)
		# print(self.p.ent_num)
		# print(self.p.event_num)
		# print(self.edge_index.shape)

		bs = sub.shape[0]
		entity_emb, relation_emb = self.forward_base(sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type)
		# entity_emb = (final_num, dim)
		# print(entity_emb.shape)
		# print(relation_emb.shape)
		# sub_emb	= torch.index_select(entity_emb, 0, sub)
		# entity_emb应当直接包含final_ent_list中的每一个entity的embedding
		sub_emb = entity_emb[sub_index_list]
		rel_emb	= torch.index_select(relation_emb, 0, rel)

		return sub_emb, rel_emb, entity_emb
	
	def predict_for_kg_completion(self, sub, rel):
		entity_emb, relation_emb = self.predict_kg_base(sub, rel)
		# print(relation_emb.shape)
		# print(rel.tolist())
		sub_emb	= torch.index_select(entity_emb, 0, sub)
		rel_emb	= torch.index_select(relation_emb, 0, rel)

		return sub_emb, rel_emb, entity_emb

	def forward_for_entity_typing(self, input_ent_id, input_labels, batch_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type, predict=False):
		# entities: (batch_size)
		# labels: (batch_size)
		ent_id = input_ent_id.squeeze(1)
		labels = input_labels.squeeze(1)

		# sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type

		entities = self.forward_base(None, None, batch_ent_list, batch_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type)[0][0: len(batch_ent_list)]

		if not predict:
			logits = self.linear_2(torch.nn.Dropout(0.2)(torch.nn.ReLU()(self.linear_1(entities))))
			soft_logits = torch.softmax(logits, 1)
			loss = torch.nn.CrossEntropyLoss()(soft_logits, labels)
			return loss

		else:
			with torch.no_grad():
				logits = self.linear_2(torch.nn.Dropout(0.2)(torch.nn.ReLU()(self.linear_1(entities))))
				soft_logits = torch.softmax(logits, 1)
				eval_loss = torch.nn.CrossEntropyLoss()(soft_logits, labels)
				pred_labels = torch.argmax(soft_logits, 1)
				return eval_loss, pred_labels
	
	def forward_for_relation_typing(self, start_id, end_id, label, start_index_list, end_index_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type, predict=False):
		# start_ent_id: (batch_size, 1)
		# end_ent_id: (batch_size, 1)
		# labels: (batch_size, 1)
		start_ent_id = start_id.squeeze(1)
		end_ent_id = end_id.squeeze(1)
		labels = label.squeeze(1)

		# final_ent_list = [ _ for _ in range(self.p.ent_num)]
		# ent_neighbors = [ _ for _ in range(self.p.ent_num)]
		# rel_evt_idxs = [ _ for _ in range(self.p.event_num)]
		# evt_neighbors = [ _ for _ in range(self.p.event_num)]

		entity_emb = self.forward_base(None, None, None, None, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type)[0]

		if not predict:
			start_entities = entity_emb[start_index_list]
			end_entities = entity_emb[end_index_list]

			# start_entities = entity_emb[start_ent_id]
			# end_entities = entity_emb[end_ent_id]


			entities = torch.cat((start_entities, end_entities), 1)
			logits = self.rel_linear_2(torch.nn.Dropout(0.2)(torch.nn.ReLU()(self.rel_linear_1(entities))))
			soft_logits = torch.softmax(logits, 1)
			loss = torch.nn.CrossEntropyLoss()(soft_logits, labels)

			return loss

		else:
			with torch.no_grad():
				start_entities = entity_emb[start_index_list]
				end_entities = entity_emb[end_index_list]

				# start_entities = entity_emb[start_ent_id]
				# end_entities = entity_emb[end_ent_id]

				entities = torch.cat((start_entities, end_entities), 1)

				logits = self.rel_linear_2(torch.nn.Dropout(0.2)(torch.nn.ReLU()(self.rel_linear_1(entities))))
				soft_logits = torch.softmax(logits, 1)
				eval_loss = torch.nn.CrossEntropyLoss()(soft_logits, labels)
				pred_labels = torch.argmax(soft_logits, 1)
				# print(pred_labels)
				# print(label.squeeze(1))
				return eval_loss, pred_labels

			
class CompGCN_TransE(CompGCNBase):
	def __init__(self, event_edge_index, edge_index, edge_type, event_index, role_type, role_mask, entity_event_index, entity_mask, params=None):
		super(self.__class__, self).__init__(event_edge_index, edge_index, edge_type, event_index, role_type, role_mask, entity_event_index, entity_mask, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type):

		sub_emb, rel_emb, all_ent	= self.forward_for_kg_completion(sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type)
		obj_emb				= sub_emb + rel_emb

		x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)		
		score	= torch.sigmoid(x)

		return score

class CompGCN_DistMult(CompGCNBase):
	def __init__(self, event_edge_index, edge_index, edge_type, event_index, role_type, role_mask, entity_event_index, entity_mask, params=None):
		super(self.__class__, self).__init__(event_edge_index, edge_index, edge_type, event_index, role_type, role_mask, entity_event_index, entity_mask, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type):

		sub_emb, rel_emb, all_ent	= self.forward_for_kg_completion(sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type)
		obj_emb				= sub_emb * rel_emb

		x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score

class CompGCN_ConvE(CompGCNBase):
	def __init__(self, event_edge_index, edge_index, edge_type, event_index, role_type, role_mask, entity_event_index, entity_mask, params=None):
		# edge_index: longtensor, (2, 2*rel_num) contains start nodes and end nodes (subj, obj)
		# edge_type: longtensor, (1, 1*rel_num) contains edge types (including inverse edges)
		super(self.__class__, self).__init__(event_edge_index, edge_index, edge_type, event_index, role_type, role_mask, entity_event_index, entity_mask, params.num_rel, params)

		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type):
		# 但是输入的是一个batch的所有relation和entity，也就是说这个图并不是完整的
		# 输入的sub和rel是完全打乱了的，既包含正向relation 也包含负向relation
		# sub: torch.LongTensor(batch_size)
		# rel: torch.LongTensor(batch_size)
		sub_emb, rel_emb, all_ent	= self.forward_for_kg_completion(sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type)
		# sub_emb, rel_emb: torch.FloatTensor(batch_size, gcn_dim)
		# all_ent: torch.FloatTensor(ent_num, gcn_dim)
		# print(sub_emb.shape)
		# print(rel_emb.shape)
		# print(all_ent.shape)
		stk_inp				= self.concat(sub_emb, rel_emb)
		x				= self.bn0(stk_inp)
		x				= self.m_conv1(x)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= F.relu(x)
		# print(x.shape)
		# print(all_ent.shape)
		x = torch.mm(x, all_ent.transpose(1,0))
		# print(self.bias.shape)
		# x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score
	
	def predict(self, sub, rel):
		# 但是输入的是一个batch的所有relation和entity，也就是说这个图并不是完整的
		# 输入的sub和rel是完全打乱了的，既包含正向relation 也包含负向relation
		# sub: torch.LongTensor(batch_size)
		# rel: torch.LongTensor(batch_size)
		# sub_emb, rel_emb, all_ent	= self.forward_for_kg_completion(sub, rel, sub_index_list, init_ent_list, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors, new_event_index, new_entity_event_index, new_entity_mask, new_entity_list, new_entity_type)

		sub_emb, rel_emb, all_ent	= self.predict_for_kg_completion(sub, rel)
		# sub_emb, rel_emb: torch.FloatTensor(batch_size, gcn_dim)
		# all_ent: torch.FloatTensor(ent_num, gcn_dim)
		# print(sub_emb.shape)
		# print(rel_emb.shape)
		# print(all_ent.shape)
		stk_inp				= self.concat(sub_emb, rel_emb)
		x				= self.bn0(stk_inp)
		x				= self.m_conv1(x)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= F.relu(x)
		# print(x.shape)
		# print(all_ent.shape)
		x = torch.mm(x, all_ent.transpose(1,0))
		# print(self.bias.shape)
		# x += self.bias.expand_as(x)
		score = torch.sigmoid(x)
		# print(score)
		return score
