import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5  # neuronal threshold
lens = 0.5  # hyper-parameters of approximate function
decay = 0.2  # decay constants
num_classes = 10
batch_size = 128
learning_rate = 1e-3
num_epochs = 100  # max epoch


# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()


act_fun = ActFun.apply


# membrane potential update
def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem)  # act_fun : approximation firing function
    return mem, spike


# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 4, 1, 10, 21),
           (4, 16, 1, 11, 23),
           (16, 32, 1, 12, 25),
           (32, 64, 1, 13, 27)]
# kernel size
cfg_kernel = [300, 150, 75, 37]
# fc layer
cfg_fc = [128, 5]


# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        # in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        # self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        # in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        # self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        #
        # self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        # self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=4,
                kernel_size=21,
                stride=1,
                padding=10
            ),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=16,
                kernel_size=23,
                stride=1,
                padding=11
            ),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=25,
                stride=1,
                padding=12
            ),
            # nn.ReLU(),
            # nn.AvgPool1d(kernel_size=3, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=27,
                stride=1,
                padding=13
            ),
            # nn.ReLU()

        )
        self.fc1 = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(64 * 37, 128)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 5)
        )

    # cfg_cnn = [(1, 4, 1, 10, 21),
    #            (4, 16, 1, 11, 23),
    #            (16, 32, 1, 12, 25),
    #            (32, 64, 1, 13, 27)]
    # # kernel size
    # cfg_kernel = [21, 23, 25, 27]
    # # fc layer
    # cfg_fc = [128, 5]
    def forward(self, input, time_window=20):
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], device=device)
        c3_mem = c3_spike = torch.zeros(batch_size, cfg_cnn[2][1], cfg_kernel[2], device=device)
        c4_mem = c4_spike = torch.zeros(batch_size, cfg_cnn[3][1], cfg_kernel[3], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        for step in range(time_window):  # simulation time steps
            x = input > torch.rand(input.size(), device=device)  # prob. firing
            # print(x.shape)
            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)

            x = F.max_pool1d(c1_spike, 2)

            c2_mem, c2_spike = mem_update(self.conv2, x, c2_mem, c2_spike)

            x = F.max_pool1d(c2_spike, 2)

            c3_mem, c3_spike = mem_update(self.conv3, x, c3_mem, c3_spike)

            x = F.avg_pool1d(c3_spike, 2)

            c4_mem, c4_spike = mem_update(self.conv4, x, c4_mem, c4_spike)

            x = c4_spike.view(batch_size, -1)

            # print(x.shape)
            # print(h1_mem.shape)
            # print(h1_spike.shape)
            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            # print("after:")
            # print(h1_mem.shape)
            # print(h1_spike.shape)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / time_window
        return outputs
