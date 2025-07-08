import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from models import *
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from torch_geometric.nn import GCNConv
from sklearn.neighbors import kneighbors_graph


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(len(dims) + 1):
            if i == 0:
                self.encoder.add_module('Linear%d' % i, nn.Linear(input_dim, dims[i]))
            elif i == len(dims):
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], feature_dim))
            else:
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], dims[i]))
            self.encoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.encoder(x)


class AutoDecoder2(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        super(AutoDecoder2, self).__init__()
        self.decoder = nn.Sequential()
        dims = list(reversed(dims))
        for i in range(len(dims) + 1):
            if i == 0:
                self.decoder.add_module('Linear%d' % i, nn.Linear(feature_dim, dims[i]))
            elif i == len(dims):
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], input_dim))
            else:
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], dims[i]))
            self.decoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        return self.decoder(x)


class PGNetwork(nn.Module):
    def __init__(self, num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters,
                 data_in_channels, data_hidden_channels, data_out_channels, trans_out_channels, num_heads, anchor_size,
                 dim_low):
        super(PGNetwork, self).__init__()
        self.encoders = list()
        self.decoders = list()
        self.decoders_news = list()
        self.newSGFormers = list()
        self.tsne_drawn = False

        for idx in range(num_views):
            self.encoders.append(AutoEncoder(input_sizes[idx], dim_high_feature, dims))
            self.decoders_news.append(AutoDecoder2(input_sizes[idx], data_out_channels, dims))

            self.newSGFormers.append(
                NewSGFormer(data_in_channels, data_hidden_channels, data_out_channels,
                            data_out_channels, trans_out_channels,
                            data_out_channels, trans_out_channels,
                            num_heads, anchor_size
                            ))

        self.encoders = nn.ModuleList(self.encoders)
        # self.decoders = nn.ModuleList(self.decoders)
        self.decoders_news = nn.ModuleList(self.decoders_news)
        self.newSGFormers = nn.ModuleList(self.newSGFormers)


        self.label_learning_module2 = nn.Sequential(
            nn.Linear(data_out_channels, dim_low),
            nn.Linear(dim_low, num_clusters),
            nn.Softmax(dim=1)
        )

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(dim=0)
        p = (weight.t() / weight.sum(dim=1)).t()
        return p

    def calculate_alignment_score(self, anchor_outs):
        num_views = len(anchor_outs)
        alignment_scores = []

        for i in range(num_views):
            for j in range(i + 1, num_views):
                cosine_similarity = torch.nn.functional.cosine_similarity(
                    anchor_outs[i], anchor_outs[j], dim=-1
                )
                mean_similarity = cosine_similarity.mean().item()
                alignment_scores.append((i, j, mean_similarity))

        return alignment_scores

    def plot_tsne(self, anchor_outs, outs, num_views):
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)

        # Restrict to the first three views
        selected_views = min(3, num_views)

        # Combine all embeddings for t-SNE
        embeddings = []
        view_info = []

        for view_idx in range(selected_views):
            # Add anchor features
            embeddings.append(anchor_outs[view_idx].cpu().detach().numpy())
            view_info.extend([(view_idx, 'anchor', i) for i in range(anchor_outs[view_idx].shape[0])])

            # Add node features
            embeddings.append(outs[view_idx].cpu().detach().numpy())
            view_info.extend([(view_idx, 'node', None) for _ in range(outs[view_idx].shape[0])])

        embeddings = np.vstack(embeddings)  # Combine embeddings into a single array

        # Apply t-SNE
        tsne_results = tsne.fit_transform(embeddings)

        # Visualization setup
        fig, ax = plt.subplots(figsize=(12, 8))
        color_map = plt.cm.get_cmap('tab20', sum([anchor_out.shape[0] for anchor_out in
                                                  anchor_outs[:selected_views]]))  # Generate enough colors

        current_color_idx = 0
        current_idx = 0
        for view_idx in range(selected_views):
            # Get embeddings for the current view
            anchor_count = anchor_outs[view_idx].shape[0]
            node_count = outs[view_idx].shape[0]

            # Extract t-SNE results for anchors and nodes
            anchor_tsne = tsne_results[current_idx:current_idx + anchor_count]
            node_tsne = tsne_results[current_idx + anchor_count:current_idx + anchor_count + node_count]

            # Assign a unique color to each anchor
            anchor_colors = [color_map(current_color_idx + i) for i in range(anchor_count)]

            for anchor_idx in range(anchor_count):
                # Plot each anchor with a unique color
                ax.scatter(anchor_tsne[anchor_idx, 0], anchor_tsne[anchor_idx, 1],
                           label=f"Anchor {anchor_idx + 1} (View {view_idx + 1})",
                           color=anchor_colors[anchor_idx], alpha=0.9, s=150, edgecolors='k', marker='^')

            # Calculate closest anchor for each node
            anchor_embeddings = anchor_outs[view_idx].cpu().detach().numpy()
            node_embeddings = outs[view_idx].cpu().detach().numpy()
            distances = np.linalg.norm(node_embeddings[:, None, :] - anchor_embeddings[None, :, :], axis=2)
            closest_anchors = np.argmin(distances, axis=1)

            # Plot nodes with colors matching their closest anchor
            for node_idx, anchor_idx in enumerate(closest_anchors):
                ax.scatter(node_tsne[node_idx, 0], node_tsne[node_idx, 1],
                           color=anchor_colors[anchor_idx], alpha=0.6, s=50, marker='o')

                # Add index labels for each node
                ax.text(node_tsne[node_idx, 0], node_tsne[node_idx, 1], str(node_idx),
                        fontsize=8, color='black', ha='center', va='center')

            # Update indices and color tracker
            current_idx += anchor_count + node_count
            current_color_idx += anchor_count

        # Finalize plot
        ax.set_title("t-SNE Visualization of First Three Views with Unique Anchor Coloring")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.legend()
        plt.show()

    def forward(self, data_views, is_print_TSNE):
        lbps = list()
        dvs = list()
        features = list()
        lbps_anchor = list()
        k = 3
        q_list = list()
        masks = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_views = len(data_views)
        anchor_outs = list()
        outs = list()
        padded_outs = list()

        for idx in range(num_views):
            data_view = data_views[idx].to(device)
            mask = (data_view != 0).any(dim=1)
            masks.append(mask)


        for idx in range(num_views):
            data_view = data_views[idx].to(device)

            high_features = self.encoders[idx](data_view)
            high_features_new = high_features.clone()


            mask = masks[idx]
            vaild_high_features = high_features_new[mask]
            node_feat = vaild_high_features
            adjacency_matrix = kneighbors_graph(node_feat.detach().cpu().numpy(), k, mode='connectivity',
                                                include_self=False)
            edge_index = torch.tensor(adjacency_matrix.nonzero(), dtype=torch.long)
            data = Data(x=vaild_high_features, edge_index=edge_index)
            data = data.to(device)
            out, anchor_out = self.newSGFormers[idx](data)
            anchor_outs.append(anchor_out)
            outs.append(out)
            distances = torch.cdist(out, anchor_out, p=2)  # [num_nodes, num_anchors]
            q = 1.0 / (1.0 + distances ** 2 / 1.0)
            q = q ** ((1.0 + 1.0) / 2)
            q = q / q.sum(dim=1, keepdim=True)

            num_nodes, num_anchors = q.shape
            padded_q = torch.zeros((high_features.shape[0], num_anchors), device=q.device)
            padded_q[mask] = q
            q_list.append(padded_q)

            padded_out = torch.zeros((high_features.shape[0], out.shape[1]), device=out.device)
            padded_out[mask] = out

            padded_outs.append(padded_out)
            label_probs_new = self.label_learning_module2(out)
            anchor_probs = self.label_learning_module2(anchor_out)

            features.append(high_features)
            lbps.append(label_probs_new)
            lbps_anchor.append(anchor_probs)
            dvs.append(self.decoders_news[idx](padded_out))

        q_sum = torch.zeros_like(q_list[0])
        valid_views_count = torch.zeros(q_sum.shape[0], device=q_sum.device)

        for idx in range(num_views):
            padded_out = padded_outs[idx].clone()
            mask = masks[idx]

            for i in range(padded_out.shape[0]):
                if not mask[i]:

                    weighted_anchor_out = None
                    total_weight = 0.0

                    for other_idx in range(num_views):
                        if other_idx != idx and masks[other_idx][i]:
                            weights = q_list[other_idx][i]
                            anchors = anchor_outs[idx]

                            if weighted_anchor_out is None:
                                weighted_anchor_out = torch.matmul(weights, anchors)
                            else:
                                weighted_anchor_out += torch.matmul(weights, anchors)
                            total_weight += weights.sum().item()

                    if total_weight > 0:
                        weighted_anchor_out /= total_weight
                        padded_out[i] = weighted_anchor_out

            padded_outs[idx] = padded_out

        for idx in range(num_views):
            q_sum += q_list[idx]
            valid_views_count += masks[idx].float()

        q_mean = q_sum / valid_views_count.unsqueeze(1)

        new_q_list = []
        for idx in range(num_views):
            mask = masks[idx]
            mask_expanded = mask.unsqueeze(1)
            q_filled = torch.where(mask_expanded, q_list[idx], q_mean)
            new_q_list.append(q_filled)

        p = self.target_distribution(q_mean.detach())
        return lbps, dvs, padded_outs, anchor_outs, lbps_anchor, new_q_list, p, 0
    def compute_cost_similarity(self, A, B):
        # Compute pairwise squared Euclidean distances
        A_norm = torch.sum(A ** 2, dim=1, keepdim=True)
        B_norm = torch.sum(B ** 2, dim=1, keepdim=True)
        dist_matrix = A_norm - 2 * torch.mm(A, B.T) + B_norm.T

        similarity_matrix = -torch.sqrt(torch.clamp(dist_matrix, min=0.0))

        return similarity_matrix

    def align_anchors(self, anchor_outs):
        num_views = len(anchor_outs)
        total_loss = 0.0

        for i in range(num_views):
            for j in range(i + 1, num_views):
                sim_i_j = self.compute_cost_similarity(anchor_outs[i], anchor_outs[j])
                I = torch.eye(sim_i_j.size(0), device=sim_i_j.device)
                loss = torch.nn.functional.mse_loss(sim_i_j, I)
                total_loss += loss

        return total_loss

    def sinkhorn(self, sim_matrix, max_iter=50, epsilon=1e-6):

        P = torch.exp(sim_matrix)
        for _ in range(max_iter):
            P = P / (P.sum(dim=-1, keepdim=True) + epsilon)
            P = P / (P.sum(dim=-2, keepdim=True) + epsilon)
        return P


# -----------------------------------------------------------------------------------------------------------------------

class FeedForwardNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, in_channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class AnchorToAnchorSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        # self.dropout = nn.Dropout(p=0.1)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ffn = FeedForwardNetwork(in_channels, 4 * in_channels)

    def forward(self, P):  # P: [N, in_channels]
        Qp = self.Wq(P).view(P.shape[0], self.num_heads, self.out_channels)
        Kp = self.Wk(P).view(P.shape[0], self.num_heads, self.out_channels)
        Vp = self.Wv(P).view(P.shape[0], self.num_heads, self.out_channels)

        attention_logits = torch.einsum("nhd,mhd->nhm", Qp, Kp) / math.sqrt(self.in_channels)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        # attention_weights = self.dropout(torch.softmax(attention_logits, dim=-1))
        # print(attention_weights)

        output = torch.einsum("nhm,mhd->nhd", attention_weights, Vp)
        output = output.reshape(P.shape[0], self.out_channels * self.num_heads)

        P_hat = self.norm1(P + output)
        P_tilde = self.norm2(P_hat + self.ffn(P_hat))

        return P_tilde


class AnchorToNodeCrossAttention(nn.Module):
    def __init__(self, node_in_channels, anchor_in_channels, out_channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.out_channels = out_channels
        self.in_channels = anchor_in_channels

        self.Wq_node = nn.Linear(node_in_channels, out_channels * num_heads)
        self.Wk_anchor = nn.Linear(anchor_in_channels, out_channels * num_heads)
        self.Wv_anchor = nn.Linear(anchor_in_channels, out_channels * num_heads)

        self.norm1 = nn.LayerNorm(node_in_channels)
        self.norm2 = nn.LayerNorm(node_in_channels)
        self.ffn = FeedForwardNetwork(node_in_channels, 4 * node_in_channels)

    def forward(self, H, P_tilde):
        Qh = self.Wq_node(H).reshape(-1, self.num_heads, self.out_channels)
        Kp = self.Wk_anchor(P_tilde).reshape(-1, self.num_heads, self.out_channels)
        Vp = self.Wv_anchor(P_tilde).reshape(-1, self.num_heads, self.out_channels)

        attention_logits = torch.einsum("nhd,mhd->nhm", Qh, Kp) / math.sqrt(self.in_channels)
        attention_weights = torch.softmax(attention_logits, dim=-1)

        output = torch.einsum("nhm,mhd->nhd", attention_weights, Vp)
        output = output.reshape(H.shape[0], self.out_channels * self.num_heads)
        H_hat = self.norm1(H + output)
        H_tilde = self.norm2(H_hat + self.ffn(H_hat))

        return H_tilde


class GCNnew(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNnew, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


class NewSGFormer(nn.Module):
    def __init__(self, data_in_channels, data_hidden_channels, data_out_channels,
                 anchor_in_channels, anchor_out_channels,
                 node_in_channels, node_out_channels,
                 num_heads, anchor_size):
        super().__init__()
        self.gcn = GCNnew(data_in_channels, data_hidden_channels, data_out_channels)
        self.anchor_to_anchor_attention = AnchorToAnchorSelfAttention(anchor_in_channels, anchor_out_channels,
                                                                      num_heads)
        self.anchor_to_node_attention = AnchorToNodeCrossAttention(node_in_channels, anchor_in_channels,
                                                                   node_out_channels,
                                                                   num_heads)
        self.fc = nn.Linear(node_in_channels, node_in_channels)
        self.anchor_size = anchor_size
        self.anchor_embeddings = nn.Parameter(torch.zeros(self.anchor_size, data_out_channels), requires_grad=False)

    def initialize_anchor_embeddings(self, node_embeddings):
        num_nodes = node_embeddings.size(0)
        rand_indices = torch.randperm(num_nodes)[:self.anchor_size]
        selected_anchors = node_embeddings[rand_indices]
        self.anchor_embeddings.data = selected_anchors.clone().detach()


    def forward(self, data):
        node_embeddings = self.gcn(data)
        if not self.anchor_embeddings.requires_grad:  # Only run KMeans the first time
            self.initialize_anchor_embeddings(node_embeddings)
            self.anchor_embeddings.requires_grad = True  # Allow gradients after initialization

        # anchor_embeddings = self.anchor_embeddings
        updated_anchor_embeddings = self.anchor_to_anchor_attention(self.anchor_embeddings)
        updated_node_embeddings = self.anchor_to_node_attention(node_embeddings, updated_anchor_embeddings)

        output = self.fc(updated_node_embeddings)
        anchor_out = self.fc(updated_anchor_embeddings)

        return output, anchor_out
