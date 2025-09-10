import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from layers import *
from torch_geometric.nn import GATConv

class GATBlock(nn.Module):
    def __init__(self, input_dim, nhid, nheads, alpha, num_nodes):
        super(GATBlock, self).__init__()
        
        # self.attentions = nn.ModuleList([SparseGraphAttentionLayer(input_dim, nhid, alpha) for _ in range(nheads)])
        
        
        self.attentions = GATConv(in_channels=input_dim, out_channels=nhid, heads=nheads, concat=True)
        
        
        
        # self.attentions = nn.ModuleList([
        #     GraphAttentionLayer(input_dim, nhid, alpha=alpha, concat=True) for _ in range(nheads)
        # ])
        self.selfattentions = nn.ModuleList([
            selfattention(num_nodes, nhid, num_nodes) for _ in range(nheads)
        ])
        self.prolayer = nn.Linear(nhid * nheads, nhid * nheads, bias=False)
        self.ln = nn.LayerNorm(nhid * nheads)

    def forward(self, x, adj):
        # x: [batch_size, num_nodes, input_dim]
        # adj: [batch_size, num_nodes, num_nodes]
        # print("===========================....")
        # print(x.shape)
        # out = []
        # for gat in self.attentions:
        #     # print("好几遍")
        #     # Graph attention layer expects individual node features, possibly reshape needed
        #     # print(x)
        #     # print(adj)
        #     h = gat(x, adj)      # [batch_size, num_nodes, nhid]
        #     out.append(h)
        # h_cat = torch.cat(out, dim=1)
        
        
        h_cat = self.attentions(x=x, edge_index=adj)
        
        

        h_cat = self.prolayer(h_cat)
        h_catlayer = h_cat
        temp = torch.zeros_like(h_cat)
        for selfatt in self.selfattentions:
            temp = temp+selfatt(h_cat)        
        h_cat = temp + h_catlayer
        h_ln = self.ln(h_cat)
        # print(h_ln.shape)        
        return h_ln

class MultiDeep(nn.Module):
    def __init__(self, ncell, ndrug, ncellfeat, ndrugfeat, nhid, nheads, alpha):
        """Dense version of GAT."""
        super(MultiDeep, self).__init__()
        
        self.feature_blocks_cell = nn.ModuleList()
        self.feature_blocks_cell2 = nn.ModuleList()
        
        self.feature_blocks_drug = nn.ModuleList()
        self.feature_blocks_drug2 = nn.ModuleList()

        
        for prefix, term in enumerate(zip(ncell, ndrug, ncellfeat, ndrugfeat, nhid)):
            ncell_, ndrug_, ncellfeat_, ndrugfeat_, nhid_ = term
            temp_cell = nn.ModuleList()
            temp_cell2 = nn.ModuleList()
            temp_drug = nn.ModuleList()
            temp_drug2 = nn.ModuleList()
            for prefix_, term_ in enumerate(zip(ncell_, ndrug_, ncellfeat_, ndrugfeat_, nhid_)): 
                ncell__, ndrug__, ncellfeat__, ndrugfeat__, nhid__ = term_
                
                # 第一层 GAT + SelfAttention block
                temp_cell.append(GATBlock(ncellfeat__, nhid__, nheads, alpha, ncell__))
                # 第二层 GAT + SelfAttention block
                temp_cell2.append(GATBlock(nhid__ * nheads, nhid__, nheads, alpha, ncell__))


                # 第一层 GAT + SelfAttention block
                temp_drug.append(GATBlock(ndrugfeat__, nhid__, nheads, alpha, ndrug__))
                # 第二层 GAT + SelfAttention block
                temp_drug2.append(GATBlock(nhid__ * nheads, nhid__, nheads, alpha, ndrug__))
        
            self.feature_blocks_cell.append(temp_cell)
            self.feature_blocks_cell2.append(temp_cell2)
            
            self.feature_blocks_drug.append(temp_drug)
            self.feature_blocks_drug2.append(temp_drug2)
        
        self.FClayer1 = nn.Linear((nhid[0][0] + nhid[0][1] + nhid[0][2] + nhid[1][0] + nhid[1][1] + nhid[1][2])*nheads*2, 256)
        self.FClayer2 = nn.Linear(256, 256)
        self.FClayer3 = nn.Linear(256, 1)
        self.output = nn.Sigmoid()
    
    def forward(self, cell_adj_matrix, cell_feat_matrix, cell_node_set, drug_adj_matrix, drug_feat_matrix, drug_node_set, idx_cell_drug, device):
        
        all_outputs_cell, all_outputs_drug = [], []
        
        
        cell_dim_size = cell_node_set[0][0].shape[0]
        drug_dim_size = drug_node_set[0][0].shape[0]
        for prefix, term in enumerate(zip(cell_adj_matrix, cell_feat_matrix, cell_node_set, drug_adj_matrix, drug_feat_matrix, drug_node_set)):
            cell_adj, cell_feat, cell_node, drug_adj, drug_feat, drug_node = term
            
            for prefix_, term_ in enumerate(zip(cell_adj, cell_feat, cell_node, drug_adj, drug_feat, drug_node)):
                # print("****************************************************")
                cell_adj_, cell_feat_, cell_node_, drug_adj_, drug_feat_, drug_node_ = term_
                
                cell_adj_, cell_feat_, cell_node_, drug_adj_, drug_feat_, drug_node_ = cell_adj_.to(device), cell_feat_.to(device), cell_node_.to(device), drug_adj_.to(device), drug_feat_.to(device), drug_node_.to(device)
                # print(torch.cuda.memory_summary())
                x_cell = self.feature_blocks_cell[prefix][prefix_](cell_feat_, cell_adj_)
                x_cell = self.feature_blocks_cell2[prefix][prefix_](x_cell, cell_adj_)
                # print(torch.cuda.memory_summary())
                # print("cell---------------:",prefix,prefix_)
                x_cell = x_cell.repeat_interleave(cell_node_.shape[1], dim=0)
                cell_node_ = cell_node_.view(-1,1)
                x_cell_ = scatter_mean(x_cell, cell_node_.squeeze(1), dim=0, dim_size=cell_dim_size)
                # print(torch.cuda.memory_summary())
                all_outputs_cell.append(x_cell_)
            
                x_drug = self.feature_blocks_drug[prefix][prefix_](drug_feat_, drug_adj_)
                x_drug = self.feature_blocks_drug2[prefix][prefix_](x_drug, drug_adj_)
                # print("drug---------------:",prefix,prefix_)
                x_drug = x_drug.repeat_interleave(drug_node_.shape[1], dim=0)
                drug_node_ = drug_node_.view(-1,1)
                x_drug_ = scatter_mean(x_drug, drug_node_.squeeze(1), dim=0, dim_size=drug_dim_size)
                all_outputs_drug.append(x_drug_)
        
        
        outputs_cell = torch.cat(all_outputs_cell,dim=1)
        outputs_drug = torch.cat(all_outputs_drug,dim=1)

        
        cell_drug_x = torch.cat((outputs_cell[idx_cell_drug[:, 0]], outputs_drug[idx_cell_drug[:, 1]]), dim=1)
        cell_drug_x = cell_drug_x.to(device)
        cell_drug_x = self.FClayer1(cell_drug_x)
        cell_drug_x = F.relu(cell_drug_x)
        cell_drug_x = self.FClayer2(cell_drug_x)
        cell_drug_x = F.relu(cell_drug_x)
        cell_drug_x = self.FClayer3(cell_drug_x)
        cell_drug_x = cell_drug_x.squeeze(-1)
        return cell_drug_x

