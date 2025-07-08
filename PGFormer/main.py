import argparse
import warnings
from layers import *
from loss import *
import time
import numpy as np
import torch
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PGFormer')
parser.add_argument('--load_pre_model', default=False, help='Testing if True or pre-training.')
parser.add_argument('--load_full_model', default=False, help='Testing if True or training.')
parser.add_argument('--save_model', default=True, help='Saving the model after training.')

parser.add_argument('--db', type=str, default='BDGP',
                    choices=['MNIST-USPS', 'COIL20', 'scene', 'hand', 'Fashion', 'BDGP', 'Pro'],
                    help='dataset name')
parser.add_argument('--seed', type=int, default=10, help='Initializing random seed.')
parser.add_argument("--mse_epochs", default=100, help='Number of epochs to pre-training.')
parser.add_argument("--con_epochs", default=100, help='Number of epochs to fine-tuning.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='Initializing learning rate.')
parser.add_argument('--weight_decay', type=float, default=0., help='Initializing weight decay.')
parser.add_argument("--temperature_l", type=float, default=1.0)
parser.add_argument('--batch_size', default=100, type=int,
                    help='The total number of samples must be evenly divisible by batch_size.')
parser.add_argument('--normalized', type=bool, default=False)
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx.')

args = parser.parse_args()
print("==========\nArgs:{}\n==========".format(args))

# torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    if args.db == "MNIST-USPS":
        args.learning_rate = 0.0004
        args.batch_size = 50
        args.seed = 10
        args.con_epochs = 200
        args.normalized = False

        dim_high_feature = 200
        dim_low_feature = 1024
        dims = [256, 512, 1024]
        gamma = 0.01
        alpha = 0.04

        dim_hidden_feature = 317
        dim_low = 248
        dim_out_feature = 200
        num_heads = 8
        trans_out_channels = 25

    elif args.db == "scene":
        args.learning_rate = 0.0005
        args.con_epochs = 100
        args.batch_size = 69
        args.seed = 10
        args.normalized = False

        gamma = 0.01
        alpha = 0.04

        dim_high_feature = 200
        dim_low_feature = 256
        dims = [256, 512, 1024, 2048]
        lmd = 0.01
        beta = 0.05

        dim_hidden_feature = 500
        dim_low = 100
        dim_out_feature = 200
        num_heads = 8
        trans_out_channels = 25

    elif args.db == "hand":
        args.learning_rate = 4.538807327213799e-05
        args.batch_size = 200
        args.seed = 50
        args.con_epochs = 200
        args.normalized = True

        dim_high_feature = 200
        dim_low_feature = 1024
        dims = [256, 512, 1024]

        gamma = 0.01
        alpha = 0.04

        dim_hidden_feature = 318
        dim_low = 271
        dim_out_feature = 200
        num_heads = 8
        trans_out_channels = 25

    elif args.db == "Fashion":
        args.learning_rate = 0.000323
        args.batch_size = 100
        args.con_epochs = 100
        args.seed = 20
        args.normalized = True
        args.temperature_l = 0.5

        dim_high_feature = 200
        dim_low_feature = 500
        dims = [256, 512]

        gamma = 0.01
        alpha = 0.04

        dim_hidden_feature = 427
        dim_low = 266
        dim_out_feature = 200
        num_heads = 8
        trans_out_channels = 25

    elif args.db == "BDGP":
        args.learning_rate = 0.0001585
        args.batch_size = 250
        args.seed = 10
        args.con_epochs = 200
        args.normalized = True

        dim_high_feature = 200
        dim_low_feature = 1024
        dims = [256, 512]

        gamma = 0.01
        alpha = 0.04

        dim_hidden_feature = 325
        dim_low = 176
        dim_out_feature = 200
        num_heads = 8
        trans_out_channels = 25

    elif args.db == "Pro":
        args.learning_rate = 0.0001
        args.batch_size = 50
        args.seed = 10
        args.con_epochs = 150
        args.normalized = False

        dim_high_feature = 200
        dim_low_feature = 1024
        dims = [256, 512, 1024]

        gamma = 0.01
        alpha = 0.04

        dim_hidden_feature = 320
        dim_low = 231
        dim_out_feature = 200
        num_heads = 8
        trans_out_channels = 25


set_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mv_data = MultiviewData(args.db, device)
num_views = len(mv_data.data_views)
num_samples = mv_data.labels.size
num_clusters = np.unique(mv_data.labels).size
input_sizes = np.zeros(num_views, dtype=int)
for idx in range(num_views):
    input_sizes[idx] = mv_data.data_views[idx].shape[1]

t = time.time()
is_print_TSNE = False
epochs = range(args.con_epochs)

total_loss_values = []

# neural network architecture
mnw = PGNetwork(num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters,
                  dim_high_feature, dim_hidden_feature, dim_out_feature, trans_out_channels, num_heads, num_clusters, dim_low)
# filling it into GPU
mnw = mnw.to(device)

mvc_loss = DeepMVCLoss(args.batch_size, num_clusters)
optimizer = torch.optim.Adam(mnw.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

if args.load_full_model: # perform inference
    state_dict = torch.load('./models/PG_pytorch_full_model_%s.pth' % args.db)
    mnw.load_state_dict(state_dict)
    print("Loading full-trained model...")
    print("Staring inference...")

if args.load_pre_model:
    state_dict = torch.load('./models/PG_pytorch_pre_model_%s.pth' % args.db)
    mnw.load_state_dict(state_dict)

    t = time.time()
    fine_tuning_loss_values = np.zeros(args.con_epochs, dtype=np.float64)


    forward_label_loss_values = []
    forward_prob_loss_values = []
    anchor_forward_prob_loss_values = []


    for epoch in range(args.con_epochs):
        if (epoch  == args.con_epochs - 1):
            is_print_TSNE = True
        else:
            is_print_TSNE = False

        total_loss, total_foward_label_loss, total_forward_prob_loss, total_anchor_forward_prob_loss = contrastive_train(mnw, mv_data, mvc_loss, args.batch_size, lmd, beta, gamma, alpha,
                                       args.temperature_l, args.normalized, epoch, optimizer, is_print_TSNE)
        fine_tuning_loss_values[epoch] = total_loss

        total_loss_values.append(total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss)
        forward_label_loss_values.append(total_foward_label_loss.item() if isinstance(total_foward_label_loss,
                                                                                      torch.Tensor) else total_foward_label_loss)
        forward_prob_loss_values.append(total_forward_prob_loss.item() if isinstance(total_forward_prob_loss,
                                                                                     torch.Tensor) else total_forward_prob_loss)
        anchor_forward_prob_loss_values.append(
            total_anchor_forward_prob_loss.item() if isinstance(total_anchor_forward_prob_loss,
                                                                torch.Tensor) else total_anchor_forward_prob_loss)

    plot_all_losses(epochs, total_loss_values, forward_label_loss_values, forward_prob_loss_values,
                    anchor_forward_prob_loss_values)
    print("contrastive_train finished.")
    print("Total time elapsed: {:.2f}s".format(time.time() - t))

    if args.save_model:
        torch.save(mnw.state_dict(), './models/PG_pytorch_full_model_%s.pth' % args.db)


if not args.load_full_model and not args.load_pre_model: # training from scrach

    pre_train_loss_values = pre_train(mnw, mv_data, args.batch_size, args.mse_epochs, optimizer, is_print_TSNE, num_clusters)

    if args.save_model:
        torch.save(mnw.state_dict(), './models/PG_pytorch_pre_model_%s.pth' % args.db)

    t = time.time()
    fine_tuning_loss_values = np.zeros(args.con_epochs, dtype=np.float64)

    total_loss_values = []
    forward_label_loss_values = []
    forward_prob_loss_values = []
    anchor_forward_prob_loss_values = []


    for epoch in range(args.con_epochs):
        if (epoch  == args.con_epochs - 1):
            is_print_TSNE = True
        else:
            is_print_TSNE = False

        total_loss, total_foward_label_loss, total_forward_prob_loss, total_anchor_forward_prob_loss = contrastive_train(mnw, mv_data, mvc_loss, args.batch_size, lmd, beta, gamma, alpha,
                                       args.temperature_l, args.normalized, epoch, optimizer, is_print_TSNE)
        fine_tuning_loss_values[epoch] = total_loss

        total_loss_values.append(total_loss)
        total_loss_values.append(total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss)
        forward_label_loss_values.append(total_foward_label_loss.item() if isinstance(total_foward_label_loss,
                                                                                      torch.Tensor) else total_foward_label_loss)
        forward_prob_loss_values.append(total_forward_prob_loss.item() if isinstance(total_forward_prob_loss,
                                                                                     torch.Tensor) else total_forward_prob_loss)
        anchor_forward_prob_loss_values.append(
            total_anchor_forward_prob_loss.item() if isinstance(total_anchor_forward_prob_loss,
                                                                torch.Tensor) else total_anchor_forward_prob_loss)

    print("contrastive_train finished.")
    print("Total time elapsed: {:.2f}s".format(time.time() - t))
    print(total_loss_values)

    if args.save_model:
        torch.save(mnw.state_dict(), './models/PG_pytorch_full_model_%s.pth' % args.db)



acc, nmi, pur, ari = valid(mnw, mv_data, args.batch_size, is_print_TSNE)
with open('result_%s.txt' % args.db, 'a+') as f:
    f.write('{} \t {} \t {} \t {} \t {} \t {} \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.4f} \n'.format(
        dim_high_feature, dim_low_feature, args.seed, args.batch_size,
        args.learning_rate, lmd, beta, gamma, acc, nmi, pur, (time.time() - t)))
    f.flush()


