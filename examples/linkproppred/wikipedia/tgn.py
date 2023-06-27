"""
Dynamic Link Prediction with a TGN model
Reference: 
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
"""


import timeit
import argparse
import numpy as np

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader

from torch_geometric.nn import TransformerConv

# internal imports
from tgb.utils.utils import set_random_seed
from tgb.linkproppred.evaluate import Evaluator
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import TGNMemory

from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset


overall_start = timeit.default_timer()

parser = argparse.ArgumentParser(description='parsing command line arguments as hyperparameters')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='random seed to use')
parser.parse_args()
args = parser.parse_args()
# setting random seed
seed = int(args.seed) #1,2,3,4,5
print ("setting random seed to be", seed)
torch.manual_seed(seed)
set_random_seed(seed)

LR = 0.0001
batch_size = 200
k_value = 10  # for computing metrics@k
n_epoch = 2
rnd_seed = 1234
memory_dim = time_dim = embedding_dim = 100
val_ratio = test_ratio = 0.15

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading
dataset_name = "wikipedia"
dataset = PyGLinkPropPredDataset(name=dataset_name, root="datasets")
data = dataset.get_TemporalData()
data = data.to(device)
metric = dataset.eval_metric
# split the data
train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=val_ratio, test_ratio=test_ratio
)
train_loader = TemporalDataLoader(train_data, batch_size=batch_size)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# neighhorhood sampler
neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    memory_dim,
    time_dim,
    message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=memory_dim,
    out_channels=embedding_dim,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()),
    lr=LR,
)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def train():
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test_one_vs_many(loader, neg_sampler, split_mode):
    """
    Evaluated the dynamic link prediction in an exhaustive manner
    """
    memory.eval()
    gnn.eval()
    link_pred.eval()

    torch.manual_seed(rnd_seed)  # Ensure deterministic sampling across epochs.

    mrr_list = []

    for pos_batch in loader:
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=device,
            )

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            # Get updated memory of all nodes involved in the computation.
            z, last_update = memory(n_id)
            z = gnn(
                z,
                last_update,
                edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )

            y_pred = link_pred(z[assoc[src]], z[assoc[dst]])

            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            metrics_mrr_rnk = evaluator.eval(input_dict)
            mrr_list.append(metrics_mrr_rnk[metric])

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    perf_metrics = {
        metric: float(torch.tensor(mrr_list).mean()),
    }
    return perf_metrics


print("==========================================================")
print("=================*** TGN model: ONE-VS-MANY ***===========")
print("==========================================================")

evaluator = Evaluator(name=dataset_name)

# negative sampler
NEG_SAMPLE_MODE = "hist_rnd"
neg_sampler = dataset.negative_sampler

# ==================================================== Train & Validation
# loading the validation negative samples
dataset.load_val_ns()


start_train_val = timeit.default_timer()
for epoch in range(1, n_epoch + 1):
    # training
    start_epoch_train = timeit.default_timer()
    loss = train()
    end_epoch_train = timeit.default_timer()
    print(
        f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {end_epoch_train - start_epoch_train: .4f}"
    )

    # validation
    start_val = timeit.default_timer()
    perf_metrics_val = test_one_vs_many(val_loader, neg_sampler, split_mode="val")
    end_val = timeit.default_timer()
    for perf_name, perf_value in perf_metrics_val.items():
        print(f"\tValidation: {perf_name}: {perf_value: .4f}")
    print(f"\tValidation: Elapsed time (s): {end_val - start_val: .4f}")

end_train_val = timeit.default_timer()
print(f"Train & Validation: Elapsed Time (s): {end_train_val - start_train_val: .4f}")

# ==================================================== Test
# loading the test negative samples
dataset.load_test_ns()

# testing ...
start_test = timeit.default_timer()
perf_metrics_test = test_one_vs_many(test_loader, neg_sampler, split_mode="test")
end_test = timeit.default_timer()

print(
    f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY --- NS-Mode: {NEG_SAMPLE_MODE} <<< "
)
for perf_name, perf_value in perf_metrics_test.items():
    print(f"\tTest: {perf_name}: {perf_value: .4f}")
print(f"\tTest: Elapsed Time (s): {end_test - start_test: .4f}")


overall_end = timeit.default_timer()
print(f"Overall Elapsed Time (s): {overall_end - overall_start: .4f}")
print("==============================================================")