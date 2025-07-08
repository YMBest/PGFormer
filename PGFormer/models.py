import time

import torch
import torch.nn as nn
from loss import *
from metrics import *
from dataprocessing import *
from scipy.spatial.distance import cdist

from sklearn.manifold import TSNE


def pre_train(network_model, mv_data, batch_size, epochs, optimizer, is_print_TSNE, anchor_size):
    t = time.time()
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    pre_train_loss_values = np.zeros(epochs, dtype=np.float64)

    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0.
        for batch_idx, (sub_data_views, _) in enumerate(mv_data_loader):
            _, dvs, _, _, _, _,_, _ = network_model(sub_data_views, is_print_TSNE)
            loss_list = list()
            for idx in range(num_views):
                mask = ((sub_data_views[idx] != 0)).any(dim=1).unsqueeze(1).float()

                mse_loss = criterion(dvs[idx] * mask, sub_data_views[idx] * mask)
                #loss_list.append(criterion(sub_data_views[idx], dvs[idx]))
                loss_list.append(mse_loss)
            loss = sum(loss_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        pre_train_loss_values[epoch] = total_loss
    return pre_train_loss_values

def contrastive_train(network_model, mv_data, mvc_loss, batch_size, lmd, beta, gamma, alpha, temperature_l, normalized, epoch,
                      optimizer, is_print_TSNE):

    network_model.train()
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)
    criterion = torch.nn.MSELoss()
    total_loss = 0.
    total_forward_label_loss = 0.
    total_forward_prob_loss = 0.
    total_anchor_forward_prob_loss = 0.

    for batch_idx, (sub_data_views, _) in enumerate(mv_data_loader):
        lbps, dvs, _, _, lbps_anchor, q_list, p, anchor_alignment_loss = network_model(sub_data_views, is_print_TSNE)

        loss_list = []
        forward_label_loss_list = []
        forward_prob_loss_list = []
        kl_loss = 0.  # Reset kl_loss per batch

        for i in range(num_views):

            mask = (sub_data_views[i] != 0).any(dim=1).unsqueeze(1).float()

            for j in range(i + 1, num_views):
                # Forward label loss
                f_label_loss = mvc_loss.forward_label(q_list[i], q_list[j], temperature_l, normalized)
                loss_list.append(f_label_loss)
                forward_label_loss_list.append(f_label_loss)

                # Forward probability loss
                f_prob_loss = mvc_loss.forward_prob(q_list[i], q_list[j])
                loss_list.append(f_prob_loss)
                forward_prob_loss_list.append(f_prob_loss)

            # Reconstruction loss
            #recon_loss = criterion(sub_data_views[i], dvs[i])
            recon_loss = criterion(dvs[i] * mask, sub_data_views[i] * mask)
            loss_list.append(recon_loss)

        # Compute kl_loss outside the loops over i and j
        for q in q_list:
            kl_loss += F.kl_div(q.log(), p.detach(), reduction='batchmean')
        loss_list.append(gamma * (kl_loss))
        loss_list.append(alpha * anchor_alignment_loss)

        # Sum losses
        loss = sum(loss_list)
        total_forward_label_loss += sum(forward_label_loss_list).item()
        total_forward_prob_loss += sum(forward_prob_loss_list).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss, total_forward_label_loss, total_forward_prob_loss, total_anchor_forward_prob_loss


def inference(network_model, mv_data, batch_size, is_print_TSNE):

    network_model.eval()
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)

    soft_vector = []
    pred_vectors = []
    labels_vector = []
    TSNE_features_list = []


    for v in range(num_views):
        pred_vectors.append([])

    for batch_idx, (sub_data_views, sub_labels) in enumerate(mv_data_loader):
        with torch.no_grad():
            _, _, TSNE_features, TSNE_anchors, _, lbps,_, _ = network_model(sub_data_views, is_print_TSNE)
            lbp = sum(lbps)/num_views

        for idx in range(num_views):
            pred_label = torch.argmax(lbps[idx], dim=1)
            pred_vectors[idx].extend(pred_label.detach().cpu().numpy())

        soft_vector.extend(lbp.detach().cpu().numpy())
        labels_vector.extend(sub_labels)

        #fused_features = torch.mean(torch.stack(TSNE_features), dim=0)
        fused_features = TSNE_features[0]
        TSNE_features_list.append(fused_features.cpu())


    for idx in range(num_views):
        pred_vectors[idx] = np.array(pred_vectors[idx])

    # labels_vector = np.array(labels_vector).reshape(num_samples)
    actual_num_samples = len(soft_vector)
    labels_vector = np.array(labels_vector).reshape(actual_num_samples)
    total_pred = np.argmax(np.array(soft_vector), axis=1)
    TSNE_featuress = torch.cat(TSNE_features_list, dim=0).numpy()



    return total_pred, pred_vectors, labels_vector, TSNE_featuress



def valid(network_model, mv_data, batch_size, is_print_TSNE):

    total_pred, pred_vectors, labels_vector, _= inference(network_model, mv_data, batch_size, is_print_TSNE)
    num_views = len(mv_data.data_views)

    print("Clustering results on cluster assignments of each view:")
    for idx in range(num_views):
        acc, nmi, pur, ari = calculate_metrics(labels_vector,  pred_vectors[idx])
        print('ACC{} = {:.4f} NMI{} = {:.4f} PUR{} = {:.4f} ARI{}={:.4f}'.format(idx+1, acc,
                                                                                 idx+1, nmi,
                                                                                 idx+1, pur,
                                                                                 idx+1, ari))

    print("Clustering results on semantic labels: " + str(labels_vector.shape[0]))
    acc, nmi, pur, ari = calculate_metrics(labels_vector, total_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI={:.4f}'.format(acc, nmi, pur, ari))

    return acc, nmi, pur, ari

def Get1_Top_N_Probabilities_Matrix(N, features, anchors):
    distances = cdist(features, anchors)
    distances = (distances - distances.mean()) / distances.std()
    similarities = np.exp(-distances)

    probabilities = F.softmax(torch.tensor(similarities), dim=1).numpy()

    top_indices = np.argsort(probabilities, axis=1)[:, -N:]
    top_probabilities = np.take_along_axis(probabilities, top_indices, axis=1)
    top_probabilities /= top_probabilities.sum(axis=1, keepdims=True)

    top_probabilities_full = np.zeros_like(probabilities)
    np.put_along_axis(top_probabilities_full, top_indices, top_probabilities, axis=1)
    return top_probabilities_full

import matplotlib.pyplot as plt

def plot_all_losses(epochs, total_loss, forward_label_loss, forward_prob_loss, anchor_forward_prob_loss):
    plt.figure(figsize=(10, 8))

    plt.plot(epochs, total_loss, label="Total Loss", marker='o', linestyle='-', color='r')
    plt.plot(epochs, forward_label_loss, label="Forward Label Loss", marker='o', linestyle='-', color='g')
    plt.plot(epochs, forward_prob_loss, label="Forward Prob Loss", marker='o', linestyle='-', color='b')
    plt.plot(epochs, anchor_forward_prob_loss, label="Anchor Forward Prob Loss", marker='o', linestyle='-', color='orange')

    plt.title("Loss to Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

