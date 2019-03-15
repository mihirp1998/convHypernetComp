from tensorboardX import SummaryWriter
writer = SummaryWriter()
import numpy as np
import torch
a = torch.load('checkpoint100_100vids_wn/hypernet_{}_{:08d}.pth'.format(s, epoch))

# a = np.random.randn(100,16)
writer.add_embedding(a['context_embeddings.weight'])