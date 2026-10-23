import torch
from torch.autograd import Variable
from torch.nn import functional as F
from collections import OrderedDict
from torch.func import functional_call

from embeddings import item, user


class user_preference_estimator(torch.nn.Module):
    def __init__(self, config):
        super(user_preference_estimator, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.fc1_in_dim = config['embedding_dim'] * 8
        self.fc2_in_dim = config['first_fc_hidden_dim']
        self.fc2_out_dim = config['second_fc_hidden_dim']
        self.use_cuda = config['use_cuda']

        self.item_emb = item(config)
        self.user_emb = user(config)
        self.fc1 = torch.nn.Linear(self.fc1_in_dim, self.fc2_in_dim)
        self.fc2 = torch.nn.Linear(self.fc2_in_dim, self.fc2_out_dim)
        self.linear_out = torch.nn.Linear(self.fc2_out_dim, 1)

    def forward(self, x, training = True):
        rate_idx = Variable(x[:, 0], requires_grad=False)
        genre_idx = Variable(x[:, 1:26], requires_grad=False)
        director_idx = Variable(x[:, 26:2212], requires_grad=False)
        actor_idx = Variable(x[:, 2212:10242], requires_grad=False)
        gender_idx = Variable(x[:, 10242], requires_grad=False)
        age_idx = Variable(x[:, 10243], requires_grad=False)
        occupation_idx = Variable(x[:, 10244], requires_grad=False)
        area_idx = Variable(x[:, 10245], requires_grad=False)

        item_emb = self.item_emb(rate_idx, genre_idx, director_idx, actor_idx)
        user_emb = self.user_emb(gender_idx, age_idx, occupation_idx, area_idx)
        x = torch.cat((item_emb, user_emb), 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.linear_out(x)


class MeLU(torch.nn.Module):
    def __init__(self, config):
        super(MeLU, self).__init__()
        self.use_cuda = config['use_cuda']
        self.model = user_preference_estimator(config)
        self.local_lr = config['local_lr']
        self.meta_optim = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
        self.local_update_target_weight_name = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'linear_out.weight', 'linear_out.bias']

    def forward(self, support_set_x, support_set_y, query_set_x, num_local_update):
        # recover current parameters while keeping the link with Autograd
        fast_weights = OrderedDict(self.model.named_parameters())

        for idx in range(num_local_update):
            support_set_y_pred = functional_call(self.model, fast_weights, (support_set_x,))
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            
            new_fast_weights = OrderedDict()
            for (name, param), grad in zip(fast_weights.items(), grads):
                if name in self.local_update_target_weight_name:
                    new_fast_weights[name] = param - self.local_lr * grad
                else:
                    new_fast_weights[name] = param
            fast_weights = new_fast_weights

        # evaluate the query set without modifying the model in-place
        query_set_y_pred = functional_call(self.model, fast_weights, (query_set_x,))
        return query_set_y_pred

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys, num_local_update):
        batch_sz = len(support_set_xs)
        losses_q = []
        if self.use_cuda:
            for i in range(batch_sz):
                support_set_xs[i] = support_set_xs[i].cuda()
                support_set_ys[i] = support_set_ys[i].cuda()
                query_set_xs[i] = query_set_xs[i].cuda()
                query_set_ys[i] = query_set_ys[i].cuda()
        for i in range(batch_sz):
            query_set_y_pred = self.forward(support_set_xs[i], support_set_ys[i], query_set_xs[i], num_local_update)
            loss_q = F.mse_loss(query_set_y_pred, query_set_ys[i].view(-1, 1))
            losses_q.append(loss_q)
        losses_q = torch.stack(losses_q).mean(0)
        self.meta_optim.zero_grad()
        losses_q.backward()

        # check if layers actually have gradients greater than zero
        if not hasattr(self, '_grad_checked'):
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    print(f"[ERROR] {name}: gradient None")

        self.meta_optim.step()
        return losses_q.item()

    def get_weight_avg_norm(self, support_set_x, support_set_y, num_local_update):
        tmp = 0.
        if self.use_cuda:
            support_set_x = support_set_x.cuda()
            support_set_y = support_set_y.cuda()
            
        fast_weights = OrderedDict(self.model.named_parameters())
        for idx in range(num_local_update):
            support_set_y_pred = functional_call(self.model, fast_weights, (support_set_x,))
            loss = F.mse_loss(support_set_y_pred, support_set_y.view(-1, 1))
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            
            new_fast_weights = OrderedDict()
            for (name, param), grad in zip(fast_weights.items(), grads):
                if name in self.local_update_target_weight_name:
                    new_fast_weights[name] = param - self.local_lr * grad
                else:
                    new_fast_weights[name] = param
            fast_weights = new_fast_weights

            for name in self.local_update_target_weight_name:
                tmp += torch.norm(fast_weights[name])
        return tmp / num_local_update