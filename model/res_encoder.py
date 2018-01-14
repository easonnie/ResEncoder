import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch_util
from tqdm import tqdm
import util.save_tool as save_tool
import os
from datetime import datetime

import util.data_loader as data_loader
import config
import fire


def model_eval(model, data_iter, criterion, pred=False):
    model.eval()
    data_iter.init_epoch()
    n_correct = loss = 0
    totoal_size = 0

    if not pred:
        for batch_idx, batch in enumerate(data_iter):

            s1, s1_l = batch.premise
            s2, s2_l = batch.hypothesis
            y = batch.label.data - 1

            pred = model(s1, s1_l - 1, s2, s2_l - 1)
            n_correct += (torch.max(pred, 1)[1].view(batch.label.size()).data == y).sum()

            loss += criterion(pred, batch.label - 1).data[0] * batch.batch_size
            totoal_size += batch.batch_size

        avg_acc = 100. * n_correct / totoal_size
        avg_loss = loss / totoal_size

        return avg_acc, avg_loss
    else:
        pred_list = []
        for batch_idx, batch in enumerate(data_iter):

            s1, s1_l = batch.premise
            s2, s2_l = batch.hypothesis

            pred = model(s1, s1_l - 1, s2, s2_l - 1)
            pred_list.append(torch.max(pred, 1)[1].view(batch.label.size()).data)

        return torch.cat(pred_list, dim=0)


class ResEncoder(nn.Module):
    def __init__(self, h_size=[600, 600, 600], v_size=10, d=300, mlp_d=800, dropout_r=0.1, max_l=60, k=3, n_layers=1):
        super(ResEncoder, self).__init__()
        self.Embd = nn.Embedding(v_size, d)

        self.lstm = nn.LSTM(input_size=d, hidden_size=h_size[0],
                            num_layers=1, bidirectional=True)

        self.lstm_1 = nn.LSTM(input_size=(d + h_size[0] * 2), hidden_size=h_size[1],
                              num_layers=1, bidirectional=True)

        self.lstm_2 = nn.LSTM(input_size=(d + h_size[0] * 2), hidden_size=h_size[2],
                              num_layers=1, bidirectional=True)

        self.max_l = max_l
        self.h_size = h_size
        self.k = k

        self.mlp_1 = nn.Linear(h_size[2] * 2 * 4, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.sm = nn.Linear(mlp_d, 3)

        if n_layers == 1:
            self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(dropout_r),
                                              self.sm])
        elif n_layers == 2:
            self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(dropout_r),
                                              self.mlp_2, nn.ReLU(), nn.Dropout(dropout_r),
                                              self.sm])
        else:
            print("Error num layers")

    def count_params(self):
        total_c = 0
        for param in self.parameters():
            if len(param.size()) == 2:
                d1, d2 = param.size()[0], param.size()[1]
                total_c += d1 * d2
        print("Total count:", total_c)

    def display(self):
        for param in self.parameters():
            print(param.data.size())

    def forward(self, s1, l1, s2, l2):
        if self.max_l:
            l1 = l1.clamp(max=self.max_l)
            l2 = l2.clamp(max=self.max_l)
            if s1.size(0) > self.max_l:
                s1 = s1[:self.max_l, :]
            if s2.size(0) > self.max_l:
                s2 = s2[:self.max_l, :]

        p_s1 = self.Embd(s1)
        p_s2 = self.Embd(s2)

        s1_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, p_s1, l1)
        s2_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, p_s2, l2)

        # Length truncate
        len1 = s1_layer1_out.size(0)
        len2 = s2_layer1_out.size(0)
        p_s1 = p_s1[:len1, :, :]
        p_s2 = p_s2[:len2, :, :]

        # Using high way
        s1_layer2_in = torch.cat([p_s1, s1_layer1_out], dim=2)
        s2_layer2_in = torch.cat([p_s2, s2_layer1_out], dim=2)

        s1_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1, s1_layer2_in, l1)
        s2_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1, s2_layer2_in, l2)

        s1_layer3_in = torch.cat([p_s1, s1_layer1_out + s1_layer2_out], dim=2)
        s2_layer3_in = torch.cat([p_s2, s2_layer1_out + s2_layer2_out], dim=2)

        s1_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2, s1_layer3_in, l1)
        s2_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2, s2_layer3_in, l2)

        s1_layer3_maxout = torch_util.max_along_time(s1_layer3_out, l1)
        s2_layer3_maxout = torch_util.max_along_time(s2_layer3_out, l2)

        # Only use the last layer
        features = torch.cat([s1_layer3_maxout, s2_layer3_maxout,
                              torch.abs(s1_layer3_maxout - s2_layer3_maxout),
                              s1_layer3_maxout * s2_layer3_maxout],
                             dim=1)

        out = self.classifier(features)
        return out


def train_snli():
    seed = 12
    rate = 0.1
    n_layers = 1
    mlp_d = 800
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    snli_d, mnli_d, embd = data_loader.load_data_sm(
        config.DATA_ROOT, config.EMBD_FILE, reseversed=False, batch_sizes=(32, 200, 200, 32, 32), device=0)

    s_train, s_dev, s_test = snli_d

    s_train.repeat = False

    model = ResEncoder(mlp_d=mlp_d, dropout_r=rate, n_layers=n_layers)
    model.Embd.weight.data = embd
    model.display()

    if torch.cuda.is_available():
        embd.cuda()
        model.cuda()

    start_lr = 2e-4

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
    criterion = nn.CrossEntropyLoss()

    date_now = datetime.now().strftime("%m-%d-%H:%M:%S")
    name = '[600,600,600]-3stack-bilstm-maxout-residual-{}-relu-seed({})-dr({})-mlpd({})'.format(n_layers, seed, rate, mlp_d)
    file_path = save_tool.gen_prefix(name, date_now)

    """
    Attention:!!!
        Modify this to save to log file.
    """

    save_tool.logging2file(file_path, 'code', None, __file__)

    iterations = 0
    best_dev = -1

    param_file_prefix = "{}/{}".format(file_path, "saved_params_snli")
    if not os.path.exists(os.path.join(config.ROOT_DIR, param_file_prefix)):
        os.mkdir(os.path.join(config.ROOT_DIR, param_file_prefix))

    for i in range(3):
        s_train.init_epoch()

        train_iter, dev_iter = s_train, s_dev
        train_iter.repeat = False

        if i != 0:
            SAVE_PATH = os.path.join(config.ROOT_DIR, file_path, 'm_{}_snli_e'.format(i - 1))
            model.load_state_dict(torch.load(SAVE_PATH))

        start_perf = model_eval(model, dev_iter, criterion)
        i_decay = i // 2
        lr = start_lr / (2 ** i_decay)

        epoch_start_info = "epoch:{}, learning_rate:{}, start_performance:{}/{}\n".format(i, lr, *start_perf)
        print(epoch_start_info)
        save_tool.logging2file(file_path, 'log_snli', epoch_start_info)

        for batch_idx, batch in tqdm(enumerate(train_iter)):
            iterations += 1
            model.train()

            s1, s1_l = batch.premise
            s2, s2_l = batch.hypothesis
            y = batch.label - 1

            out = model(s1, (s1_l - 1), s2, (s2_l - 1))
            loss = criterion(out, y)

            optimizer.zero_grad()

            for pg in optimizer.param_groups:
                pg['lr'] = lr

            loss.backward()
            optimizer.step()

            if i == 0:
                mod = 9000
            elif i == 1:
                mod = 1000
            else:
                mod = 100

            if (1 + batch_idx) % mod == 0:
                model.max_l = 150

                dev_score, dev_loss = model_eval(model, dev_iter, criterion)
                print('SNLI:dev:{}/{}'.format(dev_score, dev_loss), end='\n')

                model.max_l = 60

                if best_dev < dev_score:

                    best_dev = dev_score

                    now = datetime.now().strftime("%m-%d-%H:%M:%S")
                    log_info = "{}\t{}\tdev:{}/{}".format(i, iterations, dev_score, dev_loss, now)
                    save_tool.logging2file(file_path, "log_snli", log_info)

                    save_path = os.path.join(config.ROOT_DIR, param_file_prefix,
                                             'e({})_dev({})'.format(i, dev_score))

                    torch.save(model.state_dict(), save_path)

        SAVE_PATH = os.path.join(config.ROOT_DIR, file_path, 'm_{}_snli_e'.format(i))
        torch.save(model.state_dict(), SAVE_PATH)


def eval(model_path, mode='dev'):
    snli_d, mnli_d, embd = data_loader.load_data_sm(
        config.DATA_ROOT, config.EMBD_FILE, reseversed=False, batch_sizes=(32, 200, 200, 32, 32), device=0)

    s_train, s_dev, s_test = snli_d

    rate = 0.1
    n_layers = 1
    mlp_d = 800

    model = ResEncoder(mlp_d=mlp_d, dropout_r=rate, n_layers=n_layers)
    model.Embd.weight.data = embd
    # model.display()

    if torch.cuda.is_available():
        embd.cuda()
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    if mode == 'dev':
        d_iter = s_dev
    else:
        d_iter = s_test

    SAVE_PATH = model_path
    model.load_state_dict(torch.load(SAVE_PATH))
    score, loss = model_eval(model, d_iter, criterion)
    print("{} score/loss:{}/{}".format(mode, score, loss))

if __name__ == '__main__':
    # train_snli()
    # eval(model_path="/home/easonnie/projects/ResEncoder/saved_model/12-04-23:22:31_[600,600,600]-3stack-bilstm-maxout-residual-1-relu-seed(12)-dr(0.1)-mlpd(800)/saved_params_snli/e(2)_dev(87.00467384677911)", mode='dev')
    # eval('test')

    # fire.Fire()
    fire.Fire()



