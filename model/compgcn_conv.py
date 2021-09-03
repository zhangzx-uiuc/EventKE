from helper import *
from model.message_passing import MessagePassing
import inspect
import sys

class CompGCNConv(MessagePassing):
	def __init__(self, in_channels, out_channels, num_rels, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()
		# 这里num_rels的含义为 添加reverse边之前边的数量
		self.p 			= params
		self.in_channels	= in_channels
		self.out_channels	= out_channels
		self.num_rels 		= num_rels
		self.act 		= act
		self.device		= None

		self.w_loop		= get_param((in_channels, out_channels))
		self.w_in		= get_param((in_channels, out_channels))
		self.w_out		= get_param((in_channels, out_channels))
		self.w_rel 		= get_param((in_channels, out_channels))
		self.loop_rel 		= get_param((1, in_channels));

		self.drop		= torch.nn.Dropout(self.p.dropout)
		self.bn			= torch.nn.BatchNorm1d(out_channels)
		if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))
		# print(self.bias.shape)


	def forward(self, x, edge_index, edge_type, rel_embed): 
		# print(edge_type.device)
		# rel_embed的维度是（2 * rel_num, init_dim）也就是加上reverse边之后的rel_embedding
		# edge_index: longtensor (2, edge_num), 这个edge_num也包含了人工添加上去的reversed edge
		# edge_type: longtensor (edge_num), 这个edge_num也包含了人工添加上去的reversed edge
		# x: entity_init_embedding: float tensor (ent_num, init_dim)
		# print(edge_index.shape)
		# print(edge_index[:, 0:10])
		if self.device is None:
			self.device = edge_index.device

		# # print(rel_embed.shape)
		# # print(self.loop_rel.shape)
		rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
		# 这里self.loop_rel是一个embedding vector，用来代表gcn中的self-loop relation
		# 所以 concat 之后， rel_embed的维度变成了(2*init_edge_types + 1)
		num_edges = edge_index.size(1) // 2
		num_ent   = x.size(0)
		# num_edges: 不包含reverse edges 的 triples数量
		# num_ent: entity_num
		self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
		self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type[num_edges:]

		self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
		self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

		# self.in_index, self.in_type 分别代表了原有triples的nodes和relation_type (2, edge_num), (edge_num)
		# self.out_index, self.out_type 分别代表了原有triples的nodes和relation_type (2, edge_num), (edge_num)
		# self.loop_index, self.loop_type 代表了self-loop的triples (2, ent_num), (ent_num) 因为每个entity都有一个self-loop 所以边的个数为ent_num

		self.in_norm     = self.compute_norm(self.in_index,  num_ent)
		self.out_norm    = self.compute_norm(self.out_index, num_ent)

		# in_index: (2, edge_num),  x: (ent_num, ent_dim), in_type: (1, edge_num), rel_embed: (2 * init_types_num + 1, init_dim)
		in_res		= self.propagate('add', self.in_index,   x=x, edge_type=self.in_type,   rel_embed=rel_embed, edge_norm=self.in_norm, 	mode='in')
		# print(in_res.shape)
		loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, 		mode='loop')
		# print(loop_res.shape)
		out_res		= self.propagate('add', self.out_index,  x=x, edge_type=self.out_type,  rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
		# print(out_res.shape)

		# # print(in_res.shape)
		# the shapes of in_res, out_res and loop_res are all (entity_num, entity_dim)
		out		= self.drop(in_res)*(1/3) + self.drop(out_res)*(1/3) + loop_res*(1/3)

		if self.p.bias: out = out + self.bias
		out = self.bn(out)
		# # print(self.act(out).shape)
		return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]		# Ignoring the self loop inserted

	def rel_transform(self, ent_embed, rel_embed):
		if   self.p.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
		elif self.p.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
		else: raise NotImplementedError

		return trans_embed

	def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
		# x_j: (edge_num, init_dim): 注意这里并不是总的entity_embedding了，而是从总的entity_embedding里面index出来的tail_embedding
		# edge_type: (2, edge_num)
		# rel_embed: (2*init_types_num+1, init_dim)
		weight 	= getattr(self, 'w_{}'.format(mode))
		# # print(weight.shape)
		# print(rel_embed.device)
		# print(edge_type.device)
		rel_emb = torch.index_select(rel_embed, 0, edge_type)
		# print(rel_embed.device)
		# print(edge_type.device)
		# print(rel_embed.shape)
		# print(edge_type.tolist())
		# rel_emb: (edge_num, edge_embed_dim)
		# x_j: (edge_num, node_embed_dim)
		xj_rel  = self.rel_transform(x_j, rel_emb)
		out	= torch.mm(xj_rel, weight)

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

		return norm

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)

class EventTempConv(MessagePassing):
	def __init__(self, in_channels, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()
		# 这里num_rels的含义为 添加reverse边之前边的数量
		self.p 			= params
		self.in_channels	= in_channels
		self.act 		= act
		self.device		= None

		self.w_in		= get_param((in_channels, in_channels))
		self.w_out		= get_param((in_channels, in_channels))

		self.drop		= torch.nn.Dropout(self.p.dropout)
		self.bn			= torch.nn.BatchNorm1d(in_channels)

		if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(in_channels)))

	def forward(self, x, edge_index): 
		# x: (event_num, event_embeds)
		# edge_index: longtensor (2, edge_num), 这个edge_num也包含了人工添加上去的reversed edge
		if self.device is None:
			self.device = edge_index.device

		# # print(rel_embed.shape)
		# # print(self.loop_rel.shape)
		# rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)
		# 这里self.loop_rel是一个embedding vector，用来代表gcn中的self-loop relation
		# 所以 concat 之后， rel_embed的维度变成了(2*init_edge_types + 1)
		num_edges = edge_index.size(1) // 2
		num_ent   = x.size(0)
		# num_edges: 不包含reverse edges 的 triples数量
		# num_ent: entity_num
		# self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
		# self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
		# self.in_type,  self.out_type  = edge_type[:num_edges], 	 edge_type [num_edges:]

		# self.loop_index  = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
		# self.loop_type   = torch.full((num_ent,), rel_embed.size(0)-1, dtype=torch.long).to(self.device)

		# self.in_index, self.in_type 分别代表了原有triples的nodes和relation_type (2, edge_num), (edge_num)
		# self.out_index, self.out_type 分别代表了原有triples的nodes和relation_type (2, edge_num), (edge_num)
		# self.loop_index, self.loop_type 代表了self-loop的triples (2, ent_num), (ent_num) 因为每个entity都有一个self-loop 所以边的个数为ent_num
		# print(edge_index.shape)
		self.in_norm     = self.compute_norm(edge_index,  num_ent)
		# self.out_norm    = self.compute_norm(self.out_index, num_ent)
		# in_index: (2, edge_num),  x: (ent_num, ent_dim), in_type: (1, edge_num), rel_embed: (2 * init_types_num + 1, init_dim)
		in_res		= self.propagate('mean', edge_index,   x=x, edge_norm=self.in_norm, 	mode='in')
		# loop_res	= self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed, edge_norm=None, 		mode='loop')
		# out_res		= self.propagate('mean', self.out_index,  x=x, edge_norm=self.out_norm,	mode='out')
		# # print(in_res.shape)
		# the shapes of in_res, out_res and loop_res are all (entity_num, entity_dim)
		# out		= self.drop(in_res)*(1/2) + self.drop(out_res)*(1/2)
		out		= self.drop(in_res)

		if self.p.bias: out = out + self.bias
		out = x + self.p.gamma1 * self.act(self.bn(out))
		# # print(self.act(out).shape)
		return out		# Ignoring the self loop inserted

	def message(self, x_j, edge_norm, mode):
		# x_j: (edge_num, init_dim): 注意这里并不是总的entity_embedding了，而是从总的entity_embedding里面index出来的tail_embedding
		# edge_type: (2, edge_num)
		# rel_embed: (2*init_types_num+1, init_dim)
		weight 	= getattr(self, 'w_{}'.format(mode))
		# # print(weight.shape)
		# rel_emb = torch.index_select(rel_embed, 0, edge_type)
		# rel_emb: (edge_num, edge_embed_dim)
		# x_j: (edge_num, node_embed_dim)
		# xj_rel  = self.rel_transform(x_j, rel_emb)
		out	= torch.mm(x_j, weight)

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

		return norm

	def __repr__(self):
		return '{}({}, {}, num_rels={})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels)

class EventConv(MessagePassing):

	def __init__(self, in_dim, gcn_dim, role_type_dim, role_num, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()
		# 这里num_rels的含义为 添加reverse边之前边的数量
		self.p 			= params
		self.in_dim	= in_dim
		self.gcn_dim	= gcn_dim
		self.role_type_dim 		= role_type_dim
		self.act 		= act
		self.device		= None
		self.role_num = role_num

		# event-event connection
		self.event_temp_mod = EventTempConv(2 * in_dim + role_type_dim, torch.relu, self.p)

		# used for first step attention
		self.loop_trigger_embed = get_param((self.p.ent_num, in_dim))
		self.loop_role_type_embed = get_param((1, role_type_dim))

		self.null_event = get_param((1, 2*in_dim + role_type_dim))

		self.bert_linear = torch.nn.Linear(in_dim, in_dim, bias=False)
		self.entity_linear = torch.nn.Linear(in_dim, in_dim, bias=False)

		# role_type_dim = event_type_dim
		self.role_type_linear = torch.nn.Linear(role_type_dim, role_type_dim, bias=False)
		self.event_type_linear = torch.nn.Linear(role_type_dim, role_type_dim, bias=False)

		self.attn_act = torch.nn.LeakyReLU(0.2)
		self.attn_fc = torch.nn.Linear(2 * role_type_dim + 2 * in_dim, 1, bias=False)

		# used for second step attention
		# self.entity_attn_linear = torch.nn.Linear(in_dim, in_dim, bias=False)
		# self.event_attn_linear = torch.nn.Linear(2 * in_dim, in_dim, bias=False)

		self.event_final_linear = torch.nn.Linear(2 * in_dim + role_type_dim, in_dim)

		self.attn_fc_2 = torch.nn.Linear(3*in_dim + role_type_dim, 1, bias=False)


	def forward(self, new_event_index, x, event_index, role_type, role_mask, entity_event_index, entity_mask, role_type_embed, event_embed, evt_type_emb, final_ent_list, ent_neighbors, rel_evt_idxs, evt_neighbors): 
		'''
			x: (entity_num, init_dim), entity embedding
			event_embed: (event_num, init_dim), event trigger embedding
			event_type_embed: (event_type_num, init_dim), event type embedding
			role_type_embed: (role_type_num, role_type_dim), role_type embedding

			event_index: (event_num, max_role_num), event ids --- entity ids
			role_type: (event_num, max_role_num), role types for each pair
			role_mask: (event_num, max_role_num), role masks

			entity_event_index: (ent_num, max_evt_num)
			entity_mask: (ent_num, max_evt_num)

			event2type: {event_id: event_type_idx}
		'''
		involved_entity_num = len(final_ent_list)
		total_entity_num = len(ent_neighbors)

		batch_event_embed = event_embed[evt_neighbors]
		batch_event_index = event_index[evt_neighbors]
		batch_role_type = role_type[evt_neighbors]
		batch_role_mask = role_mask[evt_neighbors]
		batch_evt_type_emb = evt_type_emb[evt_neighbors]
		if self.device is None:
			self.device = event_index.device

		event_num, max_role_num = batch_event_index.shape

		# print(batch_role_type.shape)
		# print(role_type_embed.shape)

		role_type_embeds = role_type_embed[batch_role_type] # role_type_embeds: (event_num, max_role_num, role_dim)
		# print("role_type_embeds", role_type_embeds.shape)

		event_entity_embeds = x[batch_event_index] # event_entity_embeds: (event_num, max_role_num, entity_dim)
		# print("event_entity_embeds", event_entity_embeds.shape)
		event_trigger_embeds = torch.stack([batch_event_embed for _ in range(max_role_num)], 1) # trigger_embeds: (event_num, max_role_num, trigger_dim)
		event_type_embeds = torch.stack([batch_evt_type_emb for _ in range(max_role_num)], 1) # type_embeds: (event_num, max_role_num, trigger_dim)
		# print("event_trigger_embeds", event_trigger_embeds.shape)
		new_role_type_embeds = torch.nn.ReLU()(self.role_type_linear(role_type_embeds))
		new_event_type_embeds = torch.nn.ReLU()(self.event_type_linear(event_type_embeds))
		# print("role_type_embeds", role_type_embeds.shape)
		new_event_entity_embeds = self.bert_linear(event_entity_embeds)
		# print("new_event_entity_embeds", new_event_entity_embeds.shape)
		event_trigger_embeds = self.bert_linear(event_trigger_embeds)
		# print("event_trigger_embeds", event_trigger_embeds.shape)
		concated_embeds = torch.cat((new_role_type_embeds, new_event_type_embeds, new_event_entity_embeds, event_trigger_embeds), 2)
		# print("concated_embeds", concated_embeds.shape)
		output_attn = self.attn_act(self.attn_fc(concated_embeds)).squeeze(2) * batch_role_mask # output_attn: (event_num, max_role_num)
		# print("output_attn", output_attn.shape)
		attn_weights = mask_softmax(output_attn, batch_role_mask)
		# print("attn_weights", attn_weights.shape)		
		entity_info = torch.sum(attn_weights.unsqueeze(2) * torch.nn.ReLU()(self.entity_linear(event_entity_embeds)), 1) # event_num, init_dim
		# print("entity_info", entity_info.shape)
		event_reprs = torch.cat((entity_info, batch_event_embed, batch_evt_type_emb), 1) # event_dim, 2*init_dim+role_type_dim
		if self.p.use_temporal:
			event_reprs = self.event_temp_mod(event_reprs, new_event_index)
		# print("event_reprs", event_reprs.shape)
		new_event_reprs = torch.cat((event_reprs, self.null_event), 0)
		origin_event_num = entity_info.shape[0]
		max_evt_num = entity_event_index.shape[1]
		loop_ent_masks = torch.Tensor([[1.0 for _ in range(total_entity_num)]]).t().to(self.device)
		# print("loop_ent_masks", loop_ent_masks.shape)
		entity_mask = torch.cat((loop_ent_masks, entity_mask), 1)
		# print("entity_mask", entity_mask.shape)
		loop_entities = torch.LongTensor([[origin_event_num for i in range(total_entity_num)]]).t().to(self.device)
		# print("loop_entities", loop_entities.shape)
		entity_event_index = torch.cat((loop_entities, entity_event_index), 1)
		# print("entity_event_index", entity_event_index.shape)
		entity_event_embeds = new_event_reprs[entity_event_index] # (ent_num, max_evt_num+1, 2*init_dim)
		# print("entity_event_embeds", entity_event_embeds.shape)		
		entity_embeds = torch.stack([x[ent_neighbors] for _ in range(max_evt_num+1)], 1) # (ent_num, max_evt_num+1, init_dim) 7376, 180, 768
		# print("entity_embeds", entity_embeds.shape)
		embeds_for_attn = torch.cat((entity_event_embeds, entity_embeds), 2)
		# print("embeds_for_attn", embeds_for_attn.shape)
		attn_weights_2 = self.attn_act(self.attn_fc_2(embeds_for_attn)).squeeze(2) * entity_mask # (ent_num, max_evt_num+1)
		# print("attn_weights_2", attn_weights_2.shape)
		ent_attn_weights = mask_softmax(attn_weights_2, entity_mask)
		# print("ent_attn_weights", ent_attn_weights.shape)
		updated_entity = torch.sum(ent_attn_weights.unsqueeze(2) * self.event_final_linear(entity_event_embeds), 1)
		# print("updated_entity", updated_entity.shape)
		updated_x = self.p.alpha*torch.nn.ReLU()(updated_entity) + x[ent_neighbors]
		# print("updated_x", updated_x.shape)
		return updated_x









