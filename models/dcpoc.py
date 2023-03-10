import numpy as np

import torch
from torch.optim import SGD, Adam
from backbone.VAE import VAE
from backbone.VAE_MLP import VAE_MLP
from torch.utils.data import DataLoader, Dataset
from models.utils.continual_model import ContinualModel
from sklearn.metrics import roc_auc_score


class DCPOC(ContinualModel):
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, args):
        super(DCPOC, self).__init__(args)

        self.nets = []

        self.lambda2 = self.args.lambda2
        self.eps = self.args.eps
        self.embedding_dim = self.args.embedding_dim
        self.weight_decay = self.args.weight_decay
        self.lambda1 = self.args.lambda1
        self.r_inter = self.args.r_inter
        self.r_intra = self.args.r_intra
        self.kld_ratio = self.args.kld_ratio

        self.current_task = -1
        self.nc = None
        self.t_c_arr = []
        self.nf = self.args.nf
        self.isPrint = self.args.isPrint

        self.mus = []
        self.log_vars = []
        self.thresholds = []

    # initial
    def begin_il(self, dataset):

        self.nc = dataset.nc
        self.t_c_arr = dataset.t_c_arr
        for i in range(self.nc):
            if self.args.dataset == 'seq-mnist':
                # 0.98m
                net = VAE_MLP(latent_dim=self.embedding_dim, device=self.device, hidden_dims=[100, 100]).to(
                    self.device)
            elif self.args.dataset == 'seq-tinyimg':
                if self.args.featureNet != 'None':
                    # 9.96m
                    net = VAE_MLP(input_dim=1000, latent_dim=self.embedding_dim, device=self.device,
                                  hidden_dims=[800, 500], is_mnist=False).to(self.device)
                else:
                    net = VAE(in_channels=3, latent_dim=self.embedding_dim, device=self.device,
                              hidden_dims=[32, 64, 128, 256, 512]).to(self.device)

            elif self.args.dataset == 'seq-cifar10':
                if self.args.featureNet != 'None':
                    # 9.96m
                    net = VAE_MLP(input_dim=1000, latent_dim=self.embedding_dim, device=self.device,
                                  hidden_dims=[800, 500], is_mnist=False).to(self.device)
                else:
                    # 13.49 VAE  85.23M resnet
                    net = VAE(in_channels=3, latent_dim=self.embedding_dim, device=self.device,
                              hidden_dims=[self.nf, self.nf * 2, self.nf * 4, self.nf * 8]).to(self.device)
            else:
                # 9.96m
                net = VAE_MLP(input_dim=1000, latent_dim=self.embedding_dim, device=self.device,
                              hidden_dims=[800, 500], is_mnist=False).to(self.device)

            self.nets.append(
                net
            )
            self.mus.append(None)
            self.log_vars.append(None)
            self.thresholds.append(None)

    def train_model(self, dataset, train_loader):
        self.current_task += 1
        categories = self.t_c_arr[self.current_task]
        prev_categories = list(range(categories[0]))

        self.reset_train_loader(train_loader, prev_categories)
        print('==========\t task: %d\t categories:' % self.current_task, categories, '\t==========')
        for category in categories:
            losses = []

            for epoch in range(self.args.n_epochs):

                avg_loss, posloss_arr, negloss_arr, kldloss_arr, maxloss_arr, pseudoloss_arr = self.train_category(
                    train_loader, category, epoch)

                losses.append(avg_loss)
                if False and epoch == 0 or (epoch + 1) % 10 == 0:
                    avg_maxloss = 0.0
                    avg_pseudoloss = 0.0
                    if self.current_task > 1:
                        avg_maxloss = np.mean(maxloss_arr)
                    if self.lambda2 != 0:
                        avg_pseudoloss = np.mean(pseudoloss_arr)
                    print(
                        "epoch: %d\t task: %d \t category: %d \t loss: %f \t posloss: %f \t negloss: %f \t kldloss: %f \t maxloss: %f \t pseudoloss: %f" % (
                            epoch + 1, self.current_task, category, avg_loss, np.mean(posloss_arr),
                            np.mean(negloss_arr), np.mean(kldloss_arr), avg_maxloss, avg_pseudoloss))


    def reset_train_loader(self, train_loader, prev_categories):

        dataset = train_loader.dataset
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        prev_dists = []
        features = []

        print('reset dataset with prev_categories', prev_categories)
        for i, data in enumerate(loader):
            input = data[0].to(self.device)

            with torch.no_grad():
                if len(prev_categories) > 0:
                    _, prev_dist = self.predict(input, prev_categories)
                    prev_dists.append(prev_dist.detach().cpu())

        if len(prev_categories) > 0:
            prev_dists = torch.cat(prev_dists, dim=0)
            # dataset.set_prevdist(prev_dists)
            dataset.set_att("prev_dists", prev_dists)

    def train_category(self, data_loader, category: int, epoch_id):

        network = self.nets[category].to(self.device)
        network.train()

        optimizer = Adam(network.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)
        avg_loss = 0.0
        sample_num = 0

        posloss_arr = []
        negloss_arr = []
        kldloss_arr = []
        maxloss_arr = []
        pseudoloss_arr = []

        categories = self.t_c_arr[self.current_task]
        prev_categories = list(range(categories[0]))
        for i, data in enumerate(data_loader):
            inputs = data[0].to(self.device)
            labels = data[1].to(self.device)

            recons, _, mu, log_var = network(inputs)

            input_flat = inputs.view(inputs.shape[0], -1)
            recons_flat = recons.view(recons.shape[0], -1)
            dist = torch.sum((input_flat - recons_flat) ** 2, dim=1)

            pos_loss = torch.relu(dist[labels == category] - self.r_intra)
            posloss_arr.append(pos_loss.detach().cpu().data.numpy())
            pos_loss_mean = torch.mean(pos_loss)

            neg_loss = self.lambda2 * torch.relu(self.r_inter - dist[labels != category])
            negloss_arr.append(neg_loss.detach().cpu().data.numpy())
            neg_loss_mean = torch.mean(neg_loss)

            mu_pos = mu[labels == category]
            log_var_pos = log_var[labels == category]
            kld_loss = self.kld_ratio * -0.5 * torch.sum(1 + log_var_pos - mu_pos ** 2 - log_var_pos.exp(), dim=1)
            kldloss_arr.append(kld_loss.detach().cpu().data.numpy())
            kld_loss_mean = torch.mean(kld_loss, dim=0)

            if self.current_task > 1:
                prev_dists = data[2].to(self.device)

                max_scores = torch.relu(dist[labels == category].view(-1, 1) - prev_dists[labels == category])
                max_loss = torch.sum(max_scores, dim=1) * self.lambda1 / len(prev_categories)
                maxloss_arr.append(max_loss.detach().cpu().data.numpy())
                max_loss_mean = torch.mean(max_loss)

                if False:
                    pseudo_input = []
                    for p_c in prev_categories:
                        p_net = self.nets[p_c]
                        p_input = p_net.sample(4)
                        pseudo_input.append(p_input)
                    pseudo_input = torch.cat(pseudo_input)
                    pseudo_recons, _, mu_p, log_var_p = network(pseudo_input)
                    pseudo_input_flat = pseudo_input.view(pseudo_input.shape[0], -1)
                    pseudo_recons_flat = pseudo_recons.view(pseudo_recons.shape[0], -1)

                    pseudo_dist = torch.sum((pseudo_input_flat - pseudo_recons_flat) ** 2, dim=1)
                    pseudo_loss = self.lambda2 * torch.relu(self.r_inter - pseudo_dist)
                    pseudoloss_arr.append(pseudo_loss.detach().cpu().data.numpy())
                    pseudo_loss_mean = torch.mean(pseudo_loss)
                else:
                    pseudo_loss_mean = 0

                loss = pos_loss_mean + neg_loss_mean + kld_loss_mean + max_loss_mean + pseudo_loss_mean

            else:
                loss = pos_loss_mean + neg_loss_mean + kld_loss_mean

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            sample_num += inputs.shape[0]

        avg_loss /= sample_num
        posloss_arr = np.hstack(posloss_arr)
        negloss_arr = np.hstack(negloss_arr)
        kldloss_arr = np.hstack(kldloss_arr)
        if len(maxloss_arr) > 0:
            maxloss_arr = np.hstack(maxloss_arr)
        if len(pseudoloss_arr) > 0:
            pseudoloss_arr = np.hstack(pseudoloss_arr)
        return avg_loss, posloss_arr, negloss_arr, kldloss_arr, maxloss_arr, pseudoloss_arr

    def reset_mu(self, loader, category):
        network = self.nets[category].to(self.device)
        network.eval()

        mu_arr = []
        log_var_arr = []
        dists = []
        with torch.no_grad():
            for i, data in enumerate(loader):
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)
                prev_dists = data[2].to(self.device)

                # inputs = data[3].to(self.device)
                # inputs = self.feat_net(inputs)

                recons, _, mu, log_var = network(inputs)

                mu_arr.append(mu[labels == category])
                log_var_arr.append(log_var[labels == category])

                # input_flat = inputs.view(inputs.shape[0], -1)
                # recons_flat = recons.view(recons.shape[0], -1)
                # dist = torch.sum((input_flat - recons_flat) ** 2, dim=1)[labels == category]
                # dists += list(dist.cpu().data.numpy().tolist())

            mu_arr = torch.cat(mu_arr)
            log_var_arr = torch.cat(log_var_arr)

            mu_mean = torch.mean(mu_arr, dim=0)
            log_var_mean = torch.mean(log_var_arr, dim=0)

            self.mus[category] = mu_mean
            self.log_vars[category] = log_var_mean

        # dists.sort()
        # self.thresholds[category] = dists[int(len(dists) * (1 - 0.9))]
        # print("threshold:", self.thresholds[category])

    def get_score(self, dist, category):
        score = 1 / (dist + 1e-6)

        return score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        categories = list(range(self.t_c_arr[self.current_task][-1] + 1))
        return self.predict(x, categories)[0]

    def predict(self, inputs: torch.Tensor, categories):
        inputs = inputs.to(self.device)
        # inputs = self.feat_net(inputs)
        outcome, dists = [], []
        with torch.no_grad():
            for i in categories:
                net = self.nets[i]
                net.to(self.device)
                net.eval()

                recons, _, mu, log_var = net(inputs)
                input_flat = inputs.view(inputs.shape[0], -1)
                recons_flat = recons.view(recons.shape[0], -1)
                dist = torch.sum((input_flat - recons_flat) ** 2, dim=1)

                scores = self.get_score(dist, i)

                outcome.append(scores.view(-1, 1))
                dists.append(dist.view(-1, 1))

        outcome = torch.cat(outcome, dim=1)
        dists = torch.cat(dists, dim=1)
        return outcome, dists

    def evaluate_aoc(self, test_loaders):

        all_outputs, all_labels = [], []
        categories = list(range(self.t_c_arr[self.current_task][-1] + 1))

        for k, test_loader in enumerate(test_loaders):
            for data in test_loader:
                inputs = data[0]
                labels = data[1]

                _, dists = self.predict(inputs, categories)

                all_outputs.append(dists.detach().cpu())
                all_labels.append(labels.detach().cpu())

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0).numpy()
        for i in range(len(categories)):
            label = np.where(all_labels == i, -1, 1)
            aoc = roc_auc_score(label, all_outputs[:, i].detach().cpu().numpy())
            print('类别', i, 'AOC:', aoc)

    def evaluate_4case(self, test_loaders):

        sample_num = 0
        correct = 0
        case_1 = 0
        case_2 = 0
        case_4 = 0

        categories = list(range(self.t_c_arr[self.current_task][-1] + 1))
        for k, test_loader in enumerate(test_loaders):
            for data in test_loader:
                inputs = data[0]
                labels = data[1]

                scores, dist = self.predict(inputs, categories)

                scores = scores.cpu()
                dist = dist.cpu()
                _, pred = torch.max(scores, 1)

                for i in range(inputs.shape[0]):
                    if pred[i] == labels[i]:
                        correct += 1
                    else:

                        softmax_fun = torch.nn.Softmax(dim=1)
                        probilities = softmax_fun(self.get_score(dist, None))
                        label_prob = probilities[i, labels[i]]

                        if label_prob < 0.10:
                            case_1 += 1
                        elif pred[i] < labels[i]:
                            case_2 += 1
                        else:
                            case_4 += 1
                sample_num += inputs.shape[0]

        print("++++++++++++++ evaluate_4case ++++++++++++++")
        print("correct  number:", correct, "rato", correct / sample_num)
        print("case_1  number:", case_1, "rato", case_1 / sample_num)
        print("case_2  number:", case_2, "rato", case_2 / sample_num)
        print("case_4  number:", case_4, "rato", case_4 / sample_num)
