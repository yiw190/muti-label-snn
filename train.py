from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model_utils import *
from dataset_utils import *
from loss_utils import *
from evaluate import test
EPOCH = 100
lr = 0.001

if __name__ == '__main__':

    file = 'data/processed_data.npz'
    x_train, x_test, y_train, y_test = load_data(file)
    train_loader = DataLoader(TrainSet(x_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TestSet(x_test, y_test), batch_size=64, shuffle=True)
    
    model = SNN(10).cuda()
    
    label_num = 0
    total = 0
    sum = 0
    threshold = 0.5
    #criterion = F.binary_cross_entropy_with_logits
    criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    #optimizer = optim.Adam(params=model.parameters(), lr=lr)

    weight_decay = 1e-4
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=EPOCH,
                                        pct_start=0.2)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    # lambda1 = lambda epoch: 0.5 ** (epoch // 10)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    train_loss = [] 
    for epoch in range(EPOCH):
        pbar = tqdm(train_loader, desc=f'Train Epoch {epoch}/{EPOCH}')
        model.train()
        with pbar as t:
            for batch_idx, (data, label) in enumerate(t):
                total += label.shape[0] * label.shape[1]

                data = data.reshape([-1,1,52,52]).float().cuda()
                label = label.float().cuda()
                optimizer.zero_grad()
                output = model(data)
                #output = torch.sigmoid(output)

                loss = criterion(output, label)
                # loss_t, sum = threshold_loss(Lcard, threshold, output, total, sum)
                # loss_t.requires_grad_(True)
                loss.backward(retain_graph=True)
                # loss_t.backward()
                train_loss.append(loss.item())
                optimizer.step()

                pbar.set_description(f'Train Epoch: {epoch}/{EPOCH} loss: {np.mean(train_loss)}')
                ''' if batch_idx % 100 == 0:
                    correct = torch.zeros(1).squeeze().cuda()
                    total = torch.zeros(1).squeeze().cuda()
                    output_ = torch.sigmoid(model(data))
                    pred = torch.where(output_ >= 0.5, torch.ones_like(output_), torch.zeros_like(output_))
                    correct += (pred == label).sum()
                    total += label.shape[0]*label.shape[1]
                    print("epoch: ", epoch, "batch_idx: ", batch_idx, "accuracy: ",
                            correct/total*100) '''
                scheduler.step()
                functional.reset_net(model)
        #test(criterion, epoch, optimizer)
            test_loss = []
            global best_acc
            threshold = 0.5
            correct = torch.zeros(1).squeeze().cuda()
            total = torch.zeros(1).squeeze().cuda()
            pbar = tqdm(test_loader, desc=f'Test Epoch{epoch}/{EPOCH}')
            model.eval()
            with torch.no_grad():
                with pbar as t:
                    for data, label in t:
                        batch_size = label.shape[0]
                        data = data.reshape([-1,1,52,52]).float().cuda()
                        label = label.float().cuda()
                        output = model(data)

                        #output = output.cpu()
                        #output = torch.sigmoid(output)
                        if not torch.isnan(output).any():
                            loss = criterion(output, label).item()
                            test_loss.append(loss)

                            pred = torch.where(output > threshold, torch.ones_like(output), torch.zeros_like(output))
                            # correct += (pred == label).sum()
                            total += label.shape[0]
                            i = 0
                            while i < label.shape[0]:
                                if all(label[i] == pred[i]):
                                    correct += 1

                                # if f'{pred[i].cpu().numpy().astype(int).tolist()}' in dictx:
                                #     array[dictx[f'{label[i].cpu().numpy().astype(int).tolist()}']][
                                #         dictx[f'{pred[i].cpu().numpy().astype(int).tolist()}']] += 1
                                # else:
                                #     array[dictx[f'{label[i].cpu().numpy().astype(int).tolist()}']][38] += 1
                                # i += 1

                        pbar.set_description(
                            f'Test  Epoch: {epoch}/{EPOCH} ')
                        functional.reset_net(model)
                    pred_acc = correct / total
                    print("Test Accuracy of the epoch: ", pred_acc*100)
                    print(f"\nTest Loss:{round(np.mean(test_loss), 4)}")
            if pred_acc > best_acc:
                best_acc = pred_acc
                save_model(epoch, optimizer, test_loss, pred_acc)
'''    np.save('train_loss_cnn.npy', train_loss)
    np.save('test_loss_cnn.npy', test_loss)'''