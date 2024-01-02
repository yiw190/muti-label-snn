
def test(criterion, epoch, optimizer):
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

                        if f'{pred[i].cpu().numpy().astype(int).tolist()}' in dictx:
                            array[dictx[f'{label[i].cpu().numpy().astype(int).tolist()}']][
                                dictx[f'{pred[i].cpu().numpy().astype(int).tolist()}']] += 1
                        else:
                            array[dictx[f'{label[i].cpu().numpy().astype(int).tolist()}']][38] += 1
                        i += 1

                pbar.set_description(
                    f'Test  Epoch: {epoch}/{EPOCH} ')
                functional.reset_net(model)
            pred_acc = correct / total
            print("Test Accuracy of the epoch: ", pred_acc*100)
            print(f"\nTest Loss:{round(np.mean(test_loss), 4)}")
    if pred_acc > best_acc:
        best_acc = pred_acc
        save_model(epoch, optimizer, test_loss, pred_acc)