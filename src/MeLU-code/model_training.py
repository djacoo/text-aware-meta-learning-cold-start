import torch
import random
import numpy as np

from options import config


def training(melu, total_dataset, batch_size, num_epoch, model_save=True, model_filename=None):
    if config['use_cuda']:
        melu.cuda()

    training_set_size = len(total_dataset)
    melu.train()
    print("\n----TRAINING STARTED----")
    for epoch in range(num_epoch):
        loss_epoch = []
        random.shuffle(total_dataset)
        num_batch = int(training_set_size / batch_size)
        a,b,c,d = zip(*total_dataset)
        for i in range(num_batch):
            try:
                supp_xs = list(a[batch_size*i:batch_size*(i+1)])
                supp_ys = list(b[batch_size*i:batch_size*(i+1)])
                query_xs = list(c[batch_size*i:batch_size*(i+1)])
                query_ys = list(d[batch_size*i:batch_size*(i+1)])

                loss_val = melu.global_update(supp_xs, supp_ys, query_xs, query_ys, config['inner'])
                loss_epoch.append(loss_val)
            except IndexError:
                continue
        print(f"Epoch [{epoch+1}/{num_epoch}] - Meta-Training MSE Loss: {np.mean(loss_epoch):.4f}")

    if model_save:
        torch.save(melu.state_dict(), model_filename)

def evaluate(melu, eval_dataset, num_local_update):
    if config['use_cuda']:
        melu.cuda()

    melu.eval()
    mae_list, mse_list = [], []
    
    for s_x, s_y, q_x, q_y in eval_dataset:
        if melu.use_cuda:
            s_x, s_y, q_x, q_y = s_x.cuda(), s_y.cuda(), q_x.cuda(), q_y.cuda()
            
        q_pred = melu.forward(s_x, s_y, q_x, num_local_update)
        
        mae = torch.abs(q_pred.view(-1) - q_y.view(-1)).mean().item()
        mse = torch.nn.functional.mse_loss(q_pred.view(-1, 1), q_y.view(-1, 1)).item()
        mae_list.append(mae)
        mse_list.append(mse)
        
    melu.train()
    return np.mean(mae_list), np.sqrt(np.mean(mse_list))