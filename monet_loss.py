import os
import sys
script_path = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(script_path)
sys.path.append(parent_folder)

from MoNet.model.utils import batch_chamfer_distance, EMD

def cd_list(predicted_seq, gt_seq, batch_size):
    m_cd = 0.
    for i, pc in enumerate(predicted_seq):
        pc = pc.reshape(batch_size, -1, 3).contiguous()
        gt = gt_seq[i]

        loss = batch_chamfer_distance(pc.permute(0,2,1).contiguous(), gt)
        # gt: (B, 3, N)
        # pc: (B, 3, N)
        m_cd += loss

    m_cd = m_cd / len(predicted_seq)
        
    return m_cd


def emd_list(predicted_seq, gt_seq, batch_size):
    # input of EMD: (B, 3, N)
    m_emd = 0.
    for i, pc in enumerate(predicted_seq):
        pc = pc.reshape(batch_size, -1, 3).contiguous()
        gt = gt_seq[i]
        loss = EMD(pc.permute(0,2,1).contiguous(), gt)
        # gt: (B, 3, N)
        # pc: (B, 3, N)
        m_emd += loss

    m_emd = m_emd / len(predicted_seq)
        
    return m_emd
