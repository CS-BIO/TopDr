import numpy as np
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import csv
from scipy.sparse import coo_matrix

path="./data/GDSC1/"

def multiomics_data():
    RNAseq = np.genfromtxt("{}{}.csv".format(path, "GDSC1_RNAseq"), delimiter=',', skip_header=1, dtype=np.dtype(str))
    for i in range(RNAseq.shape[0]):
        for j in range(RNAseq.shape[1]):
            RNAseq[i,j]=RNAseq[i,j].replace('"', '')  
    cell=list(set(RNAseq[:, 0])) 
    RNAseq = multiomics_fusion(RNAseq, cell)
    RNAseq = scale(RNAseq)

    pca = PCA(n_components=128)
    RNAseq = pca.fit_transform(np.array(RNAseq, dtype=float))

    cell_number = len(cell)
    
    cell_adj_matrix, cell_feat_matrix, cell_node_set  = sim_graph(RNAseq, cell_number)

    fingerprint = pd.read_csv("{}{}.csv".format(path, "GDSC1_fingerprint_881dim"),  skiprows=0)
    final_fingerprint = []
    for _, row in fingerprint.iterrows(): 
        row_line = row.tolist()
        final_fingerprint.append(row_line)

    drug = fingerprint.index.tolist()      
    lablefile = csv.reader(open("{}{}.csv".format(path, "GDSC1_response"), 'r'))
    lablefile = list(lablefile)
    drug_cell_lable = []
    for i in range(1, len(lablefile)):
        drug_cell_lable.append(lablefile[i])
    drug_cell_lable = np.array(drug_cell_lable)
            
    fingerprint = np.array(final_fingerprint)
    drug = list(fingerprint[:,0])
    pca = PCA(n_components=128)
    fingerprint = pca.fit_transform(np.array(fingerprint[:, 1:], dtype=float))    
    fingerprint = np.array(fingerprint[:, 1:], dtype=float)
    drug_adj_matrix,drug_feat_matrix, drug_node_set = sim_graph(fingerprint, len(drug))
    
    sample_set=data_index(cell, drug, drug_cell_lable)
    print("drug cell lable", sample_set.shape)
    
    return cell_adj_matrix, cell_feat_matrix, cell_node_set, drug_adj_matrix, drug_feat_matrix, drug_node_set, sample_set
    

def multiomics_fusion(omics_data, cell_fusion):
    finalomics = []
    cell_index = omics_data[:,0].tolist()
    for cell in cell_fusion:
        index = cell_index.index(cell)
        finalomics.append(np.array(omics_data[index, 1:],dtype=float))
    return np.array(finalomics)


def sim_graph(omics_data, cell_number):
    sim_matrix = np.zeros((cell_number, cell_number), dtype=float)
    
    for i in range(cell_number):
        for j in range(i+1):
            sim_matrix[i,j] = np.dot(omics_data[i], omics_data[j]) / (np.linalg.norm(omics_data[i]) * np.linalg.norm(omics_data[j]))
            sim_matrix[j,i] = sim_matrix[i,j]
    
    rows_5, cols_5, values_5 = [], [], []
    rows_10, cols_10, values_10 = [], [], []
    dict_5,dict_10 = {},{}
    for i in range(cell_number):
        top_indices = np.argsort(sim_matrix[i])[-5:]
        dict_5[i] = top_indices
        for j in top_indices:
            rows_5.append(i)
            cols_5.append(j)
            values_5.append(sim_matrix[i,j])
            
        top_indices_2 = np.argsort(sim_matrix[i])[-10:]
        dict_10[i] = top_indices_2
        for k in top_indices_2:
            rows_10.append(i)
            cols_10.append(k)
            values_10.append(sim_matrix[i,k])

    adj_matrix_0_5 = coo_matrix((values_5, (rows_5, cols_5)), shape=(cell_number, cell_number))
    adj_matrix_0_5.setdiag(1)
    
    indices_0_5 = torch.tensor([adj_matrix_0_5.row, adj_matrix_0_5.col], dtype=torch.long)
    values_0_5 = torch.tensor(adj_matrix_0_5.data, dtype=torch.float)
    adj_torch_sparse_0_5 = torch.sparse_coo_tensor(indices_0_5, values_0_5, adj_matrix_0_5.shape)
    
    adj_matrix_0_10 = coo_matrix((values_10, (rows_10, cols_10)), shape=(cell_number, cell_number))
    adj_matrix_0_10.setdiag(1)
    
    indices_0_10 = torch.tensor([adj_matrix_0_10.row, adj_matrix_0_10.col], dtype=torch.long)
    values_0_10 = torch.tensor(adj_matrix_0_10.data, dtype=torch.float)
    adj_torch_sparse_0_10 = torch.sparse_coo_tensor(indices_0_10, values_0_10, adj_matrix_0_10.shape)    
    
    feat_matrix_0_5 = torch.FloatTensor(omics_data)
    node_set_0_5 = torch.LongTensor(list(range(0, cell_number))).unsqueeze(1)
    
    feat_matrix_0_10 = torch.FloatTensor(omics_data)
    node_set_0_10 = torch.LongTensor(list(range(0, cell_number))).unsqueeze(1)

    adj_matrix_1_5, feat_matrix_1_5, node_set_1_5 = complex_1(dict_5,sim_matrix)
    adj_matrix_1_10, feat_matrix_1_10, node_set_1_10  = complex_1(dict_10,sim_matrix)
    
    adj_matrix_2_5, feat_matrix_2_5, node_set_2_5  = complex_2(dict_5,sim_matrix)
    adj_matrix_2_10, feat_matrix_2_10, node_set_2_10  = complex_2(dict_10,sim_matrix)    
    
    adj_matrix = [[adj_torch_sparse_0_5,adj_matrix_1_5, adj_matrix_2_5],[adj_torch_sparse_0_10, adj_matrix_1_10, adj_matrix_2_10]]
    feat_matrix = [[feat_matrix_0_5, feat_matrix_1_5, feat_matrix_2_5],[feat_matrix_0_10, feat_matrix_1_10, feat_matrix_2_10]]
    node_set = [[node_set_0_5, node_set_1_5, node_set_2_5],[node_set_0_10, node_set_1_10, node_set_2_10]]   
    
    return adj_matrix,feat_matrix, node_set 

def data_index(cell, drug, lable):
    Reglable = np.array(lable[:,3], dtype = float)
    sample_set = []
    
    for i in range(len(lable)):  # len(lable)
        sample_set.append([cell.index(lable[i,1]),drug.index(lable[i,2]), Reglable[i]])
    print("the number of cell and drug:",len(set(np.array(sample_set)[:,0])), len(set(np.array(sample_set)[:,1])))
    sample_set = torch.Tensor(sample_set)
    return sample_set

def complex_1(dict_adj, sim_mat):
    rows, cols, edge_weights = [], [], []
    node_set = []
    
    for src, dst_list in dict_adj.items():
        for dst in dst_list:
            if dst > src:
                rows.append(src)
                cols.append(dst)
                edge_weights.append(1/sim_mat[src, dst])
                node_set.append([src, dst])
    
    num_nodes = len(node_set)
    adj_sparse = coo_matrix((edge_weights, (rows, cols)), 
                          shape=(num_nodes, num_nodes))
    
    adj_sparse.setdiag(1)
    indices = torch.tensor([adj_sparse.row, adj_sparse.col], dtype=torch.long)
    values = torch.tensor(adj_sparse.data, dtype=torch.float)
    adj_torch_sparse = torch.sparse_coo_tensor(indices, values, adj_sparse.shape)
    
    centers = np.arange(0, 10, 0.1)
    feat_matrix = np.zeros((num_nodes, len(centers)))
    for i, (src, dst) in enumerate(node_set):
        d = 1/sim_mat[src, dst]
        feat_matrix[i] = np.exp(-1.0 * (d - centers)**2)
    
    return adj_torch_sparse, torch.FloatTensor(feat_matrix), torch.LongTensor(node_set)

def complex_2(dict_adj, sim_mat):
    rows, cols, edge_weights = [], [], []
    node_set = []
    
    for src, dst_list in dict_adj.items():
        for dst in dst_list:
            if dst > src:
                for common in set(dict_adj[src]) & set(dict_adj[dst]):
                    if common > dst:
                        triangle_id = len(node_set)
                        node_set.append([src, dst, common])
                        
                        for other_tri_id, other_tri in enumerate(node_set):
                            if len(set(node_set[triangle_id]) & set(other_tri)) >= 2:
                                rows.append(triangle_id)
                                cols.append(other_tri_id)
                                edge_weights.append(1.0)  # 无权连接
    
    num_nodes = len(node_set)
    adj_sparse = coo_matrix((edge_weights, (rows, cols)),
                          shape=(num_nodes, num_nodes))
    
    adj_sparse.setdiag(1)
    
    indices = torch.tensor([adj_sparse.row, adj_sparse.col], dtype=torch.long)
    values = torch.tensor(adj_sparse.data, dtype=torch.float)
    adj_torch_sparse = torch.sparse_coo_tensor(indices, values, adj_sparse.shape)
    
    centers = np.arange(0, 4, 0.1)
    feat_matrix = np.zeros((num_nodes, 3 * len(centers)))
    for i, (src, dst, common) in enumerate(node_set):
        d1 = 1/sim_mat[src, dst]
        d2 = 1/sim_mat[src, common]
        d3 = 1/sim_mat[dst, common]
        feat_matrix[i] = np.concatenate([
            np.exp(-1.0 * (d1 - centers)**2),
            np.exp(-1.0 * (d2 - centers)**2),
            np.exp(-1.0 * (d3 - centers)**2)
        ])
    return adj_torch_sparse, torch.FloatTensor(feat_matrix), torch.LongTensor(node_set)             
                
    
    
    






















