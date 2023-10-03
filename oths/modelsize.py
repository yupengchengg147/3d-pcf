import torch
import torch.nn as nn
from models.mo_witht import PC_MO_KNN_witht

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size(model):
    params = count_parameters(model)
    size_bytes = params * 4  # 4 bytes per float32 parameter
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

if __name__ == "__main__":
    att_args = {'num_heads': 8, 'num_neighs': 40}
    model = PC_MO_KNN_witht(att_args)
    
    print(model_size(model))
    print("encoder size:" , model_size(model.sa1)+model_size(model.sa2)+model_size(model.sa3))
    print("motion size:", model_size(model.local_gat1) + model_size(model.local_gat2) + model_size(model.local_gat3))
    print("seq size: ", model_size(model.seq_att1) + model_size(model.seq_att2) + model_size(model.seq_att3))
    print("decoder size: ", model_size(model.fp10) + model_size(model.fp21) + model_size(model.fp32)
    + model_size(model.classifier1) + model_size(model.classifier2) + model_size(model.classifier3))

    # for name, param in model.named_parameters():
    #     print(name, param.shape)
