import numpy as np
import torch 
import torch.nn as  nn
from spikingjelly.clock_driven import neuron, functional, surrogate, layer


class selfattention2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 3, stride = 1,padding ='same')
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size = 3, stride = 1,padding ='same')
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1,padding ='same')
        self.gamma = nn.Parameter(torch.zeros(1))  
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, input):
        batch_size, channels, height, width = input.shape
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(input).view(batch_size, -1, height * width)
        v = self.value(input).view(batch_size, -1, height * width)
        attn_matrix = torch.bmm(q, k)  
        attn_matrix = self.softmax(attn_matrix)
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1)) 
        out = out.view(*input.shape)
 
        return self.gamma * out + input
    
class SNN(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

        self.static_conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            #nn.MaxPool2d(2),  # 60

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2)   # 30
        )
        self.cSE2 = selfattention2d(64)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(128),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2)  # 15
        )
        self.cSE3 = selfattention2d(128)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(256),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2)  # 7
        )
        
        self.cSE4 = selfattention2d(256)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(256),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
 
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256 * 2, bias=False),
            nn.Dropout(0.2),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()),
            nn.Linear(256 * 2, 8, bias=False),
            #neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()),
        )
    def forward(self, x):
        s_x = self.static_conv1(x)
        #print(x.shape)
        d_x = self.conv2(s_x)
        d_x = self.cSE2(d_x)
        d_x = self.conv3(d_x)
        d_x = self.cSE3(d_x)
        d_x = self.conv4(d_x)
        d_x = self.cSE4(d_x)
        d_x = self.conv5(d_x)
        #print(d_x_5.shape)
        out_spikes_counter = self.fc(d_x)
        

        for t in range(1, self.T):
            d_x = self.conv2(s_x)
            d_x = self.cSE2(d_x)
            d_x = self.conv3(d_x)
            d_x = self.cSE3(d_x)
            d_x = self.conv4(d_x)
            d_x = self.cSE4(d_x)
            d_x = self.conv5(d_x)
            out_spikes_counter += self.fc(d_x)

        return out_spikes_counter / self.T
    

def save_model(epoch, optimizer, average_loss, pred_acc):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': round(np.mean(average_loss), 2)
    }, '/content/drive/MyDrive/Colab Notebooks' + f'/baseline0.001_Epoch-{epoch}-Test_loss-{round(np.mean(average_loss), 4)}-{pred_acc*100}%.pth')
    print(f"\nBest accuracy:{pred_acc*100}%" )