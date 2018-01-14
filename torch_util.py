import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def pad_1d(t, pad_l):
    l = t.size(0)
    if l >= pad_l:
        return t
    else:
        pad_seq = Variable(t.data.new(pad_l - l, *t.size()[1:]).zero_())
        return torch.cat([t, pad_seq], dim=0)


def pad(t, length, batch_first=False):
    """
    Padding the sequence to a fixed length.
    :param t: [B * T * D] or [B * T] if batch_first else [T * B * D] or [T * B]
    :param length: [B]
    :param batch_first:
    :return:
    """
    if batch_first:
        # [B * T * D]
        if length <= t.size(1):
            return t
        else:
            batch_size = t.size(0)
            pad_seq = Variable(t.data.new(batch_size, length - t.size(1), *t.size()[2:]).zero_())
            # [B * T * D]
            return torch.cat([t, pad_seq], dim=1)
    else:
        # [T * B * D]
        if length <= t.size(0):
            return t
        else:
            return torch.cat([t, Variable(t.data.new(length - t.size(0), *t.size()[1:]).zero_())])


def batch_first2time_first(inputs):
    """
    Convert input from batch_first to time_first:
    [B * T * D] -> [T * B * D]
    
    :param inputs:
    :return:
    """
    return torch.transpose(inputs, 0, 1)


def time_first2batch_first(inputs):
    """
    Convert input from batch_first to time_first:
    [T * B * D] -> [B * T * D] 
    
    :param inputs:
    :return:
    """

    return torch.transpose(inputs, 0, 1)


def get_state_shape(rnn: nn.RNN, batch_size, bidirectional=False):
    """
    Return the state shape of a given RNN. This is helpful when you want to create a init state for RNN.

    Example:
    h0 = Variable(src_seq_p.data.new(*get_state_shape(self.encoder, 3, bidirectional)).zero_())
    
    :param rnn: 
    :param batch_size: 
    :param bidirectional: 
    :return: 
    """
    if bidirectional:
        return rnn.num_layers * 2, batch_size, rnn.hidden_size
    else:
        return rnn.num_layers, batch_size, rnn.hidden_size


def pack_list_sequence(inputs, l, batch_first=False):
    """
    Pack a batch of Tensor into one Tensor.
    :param inputs: 
    :param l: 
    :return: 
    """
    batch_list = []
    max_l = max(list(l))
    batch_size = len(inputs)

    for b_i in range(batch_size):
        batch_list.append(pad(inputs[b_i], max_l))
    pack_batch_list = torch.stack(batch_list, dim=1) if not batch_first \
        else torch.stack(batch_list, dim=0)
    return pack_batch_list


def pack_for_rnn_seq(inputs, lengths, batch_first=False):
    """
    :param inputs: [T * B * D] 
    :param lengths:  [B]
    :return: 
    """
    if not batch_first:
        _, sorted_indices = lengths.sort()
        '''
            Reverse to decreasing order
        '''
        r_index = reversed(list(sorted_indices))

        s_inputs_list = []
        lengths_list = []
        reverse_indices = np.zeros(lengths.size(0), dtype=np.int64)

        for j, i in enumerate(r_index):
            s_inputs_list.append(inputs[:, i, :].unsqueeze(1))
            lengths_list.append(lengths[i])
            reverse_indices[i] = j

        reverse_indices = list(reverse_indices)

        s_inputs = torch.cat(s_inputs_list, 1)
        packed_seq = nn.utils.rnn.pack_padded_sequence(s_inputs, lengths_list)

        return packed_seq, reverse_indices

    else:
        _, sorted_indices = lengths.sort()
        '''
            Reverse to decreasing order
        '''
        r_index = reversed(list(sorted_indices))

        s_inputs_list = []
        lengths_list = []
        reverse_indices = np.zeros(lengths.size(0), dtype=np.int64)

        for j, i in enumerate(r_index):
            s_inputs_list.append(inputs[i, :, :])
            lengths_list.append(lengths[i])
            reverse_indices[i] = j

        reverse_indices = list(reverse_indices)

        s_inputs = torch.stack(s_inputs_list, dim=0)
        # print(s_inputs)
        # print(lengths_list)
        packed_seq = nn.utils.rnn.pack_padded_sequence(s_inputs, lengths_list, batch_first=batch_first)

        return packed_seq, reverse_indices


def unpack_from_rnn_seq(packed_seq, reverse_indices, batch_first=False):
    unpacked_seq, _ = nn.utils.rnn.pad_packed_sequence(packed_seq, batch_first=batch_first)
    s_inputs_list = []

    if not batch_first:
        for i in reverse_indices:
            s_inputs_list.append(unpacked_seq[:, i, :].unsqueeze(1))
        return torch.cat(s_inputs_list, 1)
    else:
        for i in reverse_indices:
            s_inputs_list.append(unpacked_seq[i, :, :].unsqueeze(0))
        return torch.cat(s_inputs_list, 0)


def auto_rnn(rnn: nn.RNN, seqs, lengths, batch_first=True):

    batch_size = seqs.size(0) if batch_first else seqs.size(1)
    state_shape = get_state_shape(rnn, batch_size, rnn.bidirectional)

    h0 = c0 = Variable(seqs.data.new(*state_shape).zero_())

    packed_pinputs, r_index = pack_for_rnn_seq(seqs, lengths, batch_first)
    output, (hn, cn) = rnn(packed_pinputs, (h0, c0))
    output = unpack_from_rnn_seq(output, r_index, batch_first)

    return output


def pack_seqence_for_linear(inputs, lengths, batch_first=True):
    """
    :param inputs: [B * T * D] if batch_first 
    :param lengths:  [B]
    :param batch_first:  
    :param partition:  this is the new batch_size for parallel matrix process.
    :param chuck: partition the output into equal size chucks
    :return: 
    """
    batch_list = []
    if batch_first:
        for i, l in enumerate(lengths):
            batch_list.append(inputs[i, :l])
        packed_sequence = torch.cat(batch_list, 0)
        # if chuck:
        #     return list(torch.chunk(packed_sequence, chuck, dim=0))
        # else:
        return packed_sequence

    else:
        raise NotImplemented()


def chucked_forward(inputs, net, chuck=None):
    if not chuck:
        return net(inputs)
    else:
        output_list = [net(chuck) for chuck in torch.chunk(inputs, chuck, dim=0)]
        return torch.cat(output_list, dim=0)


def unpack_seqence_for_linear(inputs, lengths, batch_first=True):
    batch_list = []
    max_l = max(lengths)

    if not isinstance(inputs, list):
        inputs = [inputs]
    inputs = torch.cat(inputs)

    if batch_first:
        start = 0
        for l in lengths:
            end = start + l
            batch_list.append(pad_1d(inputs[start:end], max_l))
            start = end
        return torch.stack(batch_list)
    else:
        raise NotImplemented()


def auto_rnn_bilstm(lstm: nn.LSTM, seqs, lengths):

    batch_size = seqs.size(1)

    state_shape = lstm.num_layers * 2, batch_size, lstm.hidden_size

    h0 = c0 = Variable(seqs.data.new(*state_shape).zero_())

    packed_pinputs, r_index = pack_for_rnn_seq(seqs, lengths)

    output, (hn, cn) = lstm(packed_pinputs, (h0, c0))

    output = unpack_from_rnn_seq(output, r_index)

    return output


def auto_rnn_bigru(gru: nn.GRU, seqs, lengths):

    batch_size = seqs.size(1)

    state_shape = gru.num_layers * 2, batch_size, gru.hidden_size

    h0 = Variable(seqs.data.new(*state_shape).zero_())

    packed_pinputs, r_index = pack_for_rnn_seq(seqs, lengths)

    output, hn = gru(packed_pinputs, h0)

    output = unpack_from_rnn_seq(output, r_index)

    return output


def select_last(inputs, lengths, hidden_size):
    """
    :param inputs: [T * B * D] D = 2 * hidden_size
    :param lengths: [B]
    :param hidden_size: dimension 
    :return:  [B * D]
    """
    batch_size = inputs.size(1)
    batch_out_list = []
    for b in range(batch_size):
        batch_out_list.append(torch.cat((inputs[lengths[b] - 1, b, :hidden_size],
                                         inputs[0, b, hidden_size:])
                                        )
                              )

    out = torch.stack(batch_out_list)
    return out


def matching_matrix(s1, s2):
    """
    :param s1:  [T * B * D]
    :param s2:  [T * B * D]
    :return:  [B * T * T]
    """
    b_s1 = s1.transpose(0, 1)  # [B * T * D]
    b_s2 = s2.transpose(0, 1)  # [B * T * D]

    matrix = torch.bmm(b_s1, b_s2.transpose(1, 2))
    return matrix


def sequence_matrix_cross_alignment(s1, s2, l1, l2, matrix):
    """
    :param s1: [T * B * D]
    :param s2: [T * B * D]
    :param l1: [B]
    :param l2: [B]
    :param matrix: [B * T * T]
    :return: 
    """

    b_s1 = s1.transpose(0, 1)  # [B * T * D]
    b_s2 = s2.transpose(0, 1)  # [B * T * D]
    dim = b_s1.size(2)
    batch_size = b_s1.size(0)

    b_wsum_a2to1_list = []
    b_wsum_a1to2_list = []

    for b in range(batch_size):
        b_m = matrix[b]
        ba_s1 = b_s1[b]
        ba_s2 = b_s2[b]

        # # align s2 to s1 _a2to1
        exp_b_m_a2to1 = F.softmax(b_m[:l1[b], :l2[b]])  # [t1 * t2]

        b_weight_a2to1 = exp_b_m_a2to1.unsqueeze(2).expand(l1[b], l2[b], dim)  # [t1 * t2] -> [t1 * t2 * dim]
        b_seq_a2to1 = ba_s2[:l2[b]].unsqueeze(0).expand(l1[b], l2[b], dim)  # [t2 * dim] -> [t1 * t2 * dim]

        b_wsum_a2to1 = torch.sum(b_weight_a2to1 * b_seq_a2to1, dim=1).squeeze(1)  # [t1 * d]
        b_wsum_a2to1_list.append(b_wsum_a2to1)

        # # align s1 to s2 _a1to2
        exp_b_m_a1to2 = F.softmax(b_m[:l1[b], :l2[b]].transpose(0, 1))  # [t2 * t1]

        b_weight_a1to2 = exp_b_m_a1to2.unsqueeze(2).expand(l2[b], l1[b], dim)  # [t2 * t1] -> [t2 * t1 * d]
        b_seq_a1to2 = ba_s1[:l1[b]].unsqueeze(0).expand(l2[b], l1[b], dim)  # [t1 * dim] -> [t2 * t1 * dim]

        b_wsum_a1to2 = torch.sum(b_weight_a1to2 * b_seq_a1to2, dim=1).squeeze(1)  # [t2 * d]
        b_wsum_a1to2_list.append(b_wsum_a1to2)
        # print(b_wsum_a1to2)

    align_s2_to_s1 = pack_list_sequence(b_wsum_a2to1_list, l1)
    align_s1_to_s2 = pack_list_sequence(b_wsum_a1to2_list, l2)

    return align_s2_to_s1, align_s1_to_s2


def channel_weighted_sum(s, w, l, sharpen=None):
    batch_size = w.size(1)
    result_list = []
    for b_i in range(batch_size):
        if sharpen:
            b_w = w[:l[b_i], b_i, :] * sharpen
        else:
            b_w = w[:l[b_i], b_i, :]
        b_s = s[:l[b_i], b_i, :] # T, D
        soft_b_w = F.softmax(b_w.transpose(0, 1)).transpose(0, 1)
        # print(soft_b_w)
        # print('soft:', )
        # print(soft_b_w)
        result_list.append(torch.sum(soft_b_w * b_s, dim=0)) # [T, D] -> [1, D]
    return torch.cat(result_list, dim=0)


def topk_weighted_sum(s, w, k, l):
    batch_size = w.size(1)
    result_list = []
    for b_i in range(batch_size):
        # print(w.size())
        # print(l[b_i])
        b_w = w[:l[b_i], b_i, :]
        b_s = s[:l[b_i], b_i, :]  # T, D
        if l[b_i] == 1:
            b_topk, b_topk_indices = b_s.max(dim=0)
        elif l[b_i] < k:
            b_topk, b_topk_indices = torch.topk(b_s, l[b_i], dim=0)
        else:
            b_topk, b_topk_indices = torch.topk(b_s, k, dim=0)

        b_topk_w = torch.gather(b_w, 0, b_topk_indices)
        soft_b_topk_w = F.softmax(b_topk_w.transpose(0, 1)).transpose(0, 1)
        result_list.append(torch.sum(soft_b_topk_w * b_topk, dim=0))
    return torch.cat(result_list, dim=0)


def topk_dp_weighted_sum(s, w, l):
    batch_size = w.size(1)
    result_list = []
    for b_i in range(batch_size):
        # print(w.size())
        # print(l[b_i])

        # Dynamic pooling

        k = (int(l[b_i] - 1) // 10) + 1

        b_w = w[:l[b_i], b_i, :]
        b_s = s[:l[b_i], b_i, :]  # T, D
        if l[b_i] == 1:
            b_topk, b_topk_indices = b_s.max(dim=0)
        elif l[b_i] < k:
            b_topk, b_topk_indices = torch.topk(b_s, l[b_i], dim=0)
        else:
            b_topk, b_topk_indices = torch.topk(b_s, k, dim=0)

        b_topk_w = torch.gather(b_w, 0, b_topk_indices)
        soft_b_topk_w = F.softmax(b_topk_w.transpose(0, 1)).transpose(0, 1)
        result_list.append(torch.sum(soft_b_topk_w * b_topk, dim=0))
    return torch.cat(result_list, dim=0)


def pack_to_matching_matrix(s1, s2, cat_only=[False, False]):
    t1 = s1.size(0)
    t2 = s2.size(0)
    batch_size = s1.size(1)
    d = s1.size(2)

    expanded_p_s1 = s1.expand(t2, t1, batch_size, d)

    expanded_p_s2 = s2.view(t2, 1, batch_size, d)
    expanded_p_s2 = expanded_p_s2.expand(t2, t1, batch_size, d)

    if not cat_only[0] and not cat_only[1]:
        matrix = torch.cat((expanded_p_s1, expanded_p_s2), dim=3)
    elif not cat_only[0] and cat_only[1]:
        matrix = torch.cat((expanded_p_s1, expanded_p_s2, expanded_p_s1 * expanded_p_s2), dim=3)
    else:
        matrix = torch.cat((expanded_p_s1,
                            expanded_p_s2,
                            torch.abs(expanded_p_s1 - expanded_p_s2),
                            expanded_p_s1 * expanded_p_s2), dim=3)

    # matrix = torch.cat((expanded_p_s1,
    #                     expanded_p_s2), dim=3)

    return matrix


def max_matching(gram_matrix, l1, l2):
    batch_size = gram_matrix.size(2)
    dim = gram_matrix.size(3)
    in_d = dim // 4

    t1_seq = []
    t2_seq = []
    for b_i in range(batch_size):
        b_m = gram_matrix[:l2[b_i], :l1[b_i], b_i, :]
        max_t1_a, _ = torch.max(b_m, dim=0)
        max_t2_a, _ = torch.max(b_m, dim=1)

        t1_seq.append(max_t1_a.view(l1[b_i], -1))  # [T1, B, 4D]
        t2_seq.append(max_t2_a.view(l2[b_i], -1))  # [T2, B, 4D]

    s1_seq = pack_list_sequence(t1_seq, l1)
    s2_seq = pack_list_sequence(t2_seq, l2)
    filp_l = [s2_seq[:, :, in_d:in_d * 2], s2_seq[:, :, :in_d], s2_seq[:, :, in_d * 2:]]
    s2_seq = torch.cat(filp_l, dim=2)

    return s1_seq, s2_seq


def max_over_grammatrix(inputs, l1, l2):
    """
    :param inputs: [T2 * T1 * B * D] 
    :param l1: 
    :param l2: 
    :return: 
    """
    batch_size = inputs.size(2)
    max_out_list = []
    for b in range(batch_size):
        b_gram_matrix = inputs[:l2[b], :l1[b], b, :]
        dim = b_gram_matrix.size(-1)

        b_max, _ = torch.max(b_gram_matrix.contiguous().view(-1, dim), dim=0)

        max_out_list.append(b_max)

    max_out = torch.cat(max_out_list, dim=0)
    return max_out


def comparing_conv(matrices, l1, l2, conv_filter: nn.Linear, k_size, dropout=None,
                   padding=True, list_in=False):
    """
    :param conv_filter: [k * k * input_d] 
    :param k_size: 
    :param dropout: 
    :return: 
    """
    k = k_size

    if list_in is False:
        batch_size = matrices.size(2)
        windows = []
        for b in range(batch_size):
            b_matrix = matrices[:l2[b], :l1[b], b, :]

            if not padding:
                if l2[b] - k + 1 <= 0 or l1[b] - k + 1 <= 0:
                    raise Exception('Kernel size error k={0}, matrix=({1},{2})'.format(k, l2[b], l1[b]))

                for i in range(l2[b] - k + 1):
                    for j in range(l1[b] - k + 1):
                        window = b_matrix[i:i + k, j:j + k, :]
                        window_d = window.size(-1)
                        windows.append(window.contiguous().view(k * k * window_d))
            else:
                ch_d = b_matrix.size(-1)
                padding_n = (k - 1) // 2
                row_pad = Variable(torch.zeros(padding_n, l1[b], ch_d))

                if torch.cuda.is_available():
                    row_pad = row_pad.cuda()
                # print(b_matrix)
                # print(row_pad)
                after_row_pad = torch.cat([row_pad, b_matrix, row_pad], dim=0)
                col_pad = Variable(torch.zeros(l2[b] + 2 * padding_n, padding_n, ch_d))
                if torch.cuda.is_available():
                    col_pad = col_pad.cuda()
                after_col_pad = torch.cat([col_pad, after_row_pad, col_pad], dim=1)

                for i in range(padding_n, padding_n + l2[b]):
                    for j in range(padding_n, padding_n + l1[b]):
                        i_start = i - padding_n
                        j_start = j - padding_n
                        window = after_col_pad[i_start:i_start + k, j_start:j_start + k, :]
                        windows.append(window.contiguous().view(k * k * ch_d))

        windows = torch.stack(windows)
    else:
        batch_size = len(matrices)
        windows = []
        for b in range(batch_size):
            b_matrix = matrices[b]
            b_l2 = b_matrix.size(0)
            b_l1 = b_matrix.size(1)

            if not padding:
                if l1 is not None and l2 is not None and (l2[b] != b_l2 or l1[b] != b_l1):
                    raise Exception('Possible input matrices size error!')

                if b_l2 - k + 1 <= 0 or b_l1 - k + 1 <= 0:
                    raise Exception('Kernel size error k={0}, matrix=({1},{2})'.format(k, l2[b], l1[b]))

                for i in range(b_l2 - k + 1):
                    for j in range(b_l1 - k + 1):
                        window = b_matrix[i:i + k, j:j + k, :]
                        window_d = window.size(-1)
                        windows.append(window.contiguous().view(k * k * window_d))
            else:
                if l1 is not None and l2 is not None and (l2[b] != b_l2 or l1[b] != b_l1):
                    raise Exception('Possible input matrices size error!')

                ch_d = b_matrix.size(-1)
                padding_n = (k - 1) // 2
                row_pad = Variable(torch.zeros(padding_n, b_l1, ch_d))
                if torch.cuda.is_available():
                    row_pad = row_pad.cuda()
                after_row_pad = torch.cat([row_pad, b_matrix, row_pad], dim=0)
                col_pad = Variable(torch.zeros(b_l2 + 2 * padding_n, padding_n, ch_d))
                if torch.cuda.is_available():
                    col_pad = col_pad.cuda()
                after_col_pad = torch.cat([col_pad, after_row_pad, col_pad], dim=1)

                for i in range(padding_n, padding_n + b_l2):
                    for j in range(padding_n, padding_n + b_l1):
                        i_start = i - padding_n
                        j_start = j - padding_n
                        window = after_col_pad[i_start:i_start + k, j_start:j_start + k, :]
                        windows.append(window.contiguous().view(k * k * ch_d))

        windows = torch.stack(windows)

    if dropout:
        dropout(windows)

    # print(windows)

    out_windows = conv_filter(windows)
    a, b = torch.chunk(out_windows, 2, dim=1)
    out = a * F.sigmoid(b)

    out_list = []
    max_out_list = []
    i = 0
    for b in range(batch_size):

        if not padding:
            c_l2 = l2[b] - k + 1
            c_l1 = l1[b] - k + 1
        else:
            c_l2 = l2[b]
            c_l1 = l1[b]

        b_end = i + c_l2 * c_l1
        b_matrix = out[i:b_end, :]

        max_out, _ = b_matrix.max(dim=0)
        max_out_list.append(max_out.squeeze())

        dim = b_matrix.size(-1)
        out_list.append(b_matrix.view(c_l2, c_l1, dim))
        i = b_end

    max_out = torch.stack(max_out_list)
    # for out in out_list:
    #     max_out = torch.max(out.view(1, -1))

    return out_list, max_out


def max_along_time(inputs, lengths, list_in=False, batch_first=False):
    """
    :param inputs: [T * B * D] 
    :param lengths:  [B]
    :return: [B * D] max_along_time
    """
    ls = list(lengths)

    if not batch_first:
        if not list_in:
            b_seq_max_list = []
            for i, l in enumerate(ls):
                seq_i = inputs[:l, i, :]
                seq_i_max, _ = seq_i.max(dim=0)
                seq_i_max = seq_i_max.squeeze()
                b_seq_max_list.append(seq_i_max)

            return torch.stack(b_seq_max_list)
        else:
            b_seq_max_list = []
            for i, l in enumerate(ls):
                seq_i = inputs[i]
                seq_i_max, _ = seq_i.max(dim=0)
                seq_i_max = seq_i_max.squeeze()
                b_seq_max_list.append(seq_i_max)

            return torch.stack(b_seq_max_list)
    else:
        b_seq_max_list = []
        for i, l in enumerate(ls):
            seq_i = inputs[i, :l, :]
            seq_i_max, _ = seq_i.max(dim=0)
            seq_i_max = seq_i_max.squeeze()
            b_seq_max_list.append(seq_i_max)

        return torch.stack(b_seq_max_list)


def topk_along_time(inputs, k, lengths):
    """
    :param inputs: [T * B * D] 
    :param lengths:  [B]
    :return: [B * D] max_along_time
    """
    ls = list(lengths)
    d = inputs.size(-1)
    pad_z = Variable(inputs.data.new(1, d).zero_())

    b_seq_max_list = []
    for i, l in enumerate(ls):
        seq_i = inputs[:l, i, :]
        if l == 1:
            seq_i = torch.cat((seq_i, pad_z), dim=0)
        seq_i_topk, _ = torch.topk(seq_i, k, dim=0)
        b_seq_max_list.append(seq_i_topk.view(1, -1))

    return torch.cat(b_seq_max_list)


def topk_avg_along_time(inputs, k, lengths, list_in=False):
    ls = list(lengths)

    b_seq_max_list = []
    for i, l in enumerate(ls):
        seq_i = inputs[:l, i, :] if not list_in else inputs[i]
        if l == 1:
            seq_i_topk_avg, _ = seq_i.max(dim=0)
        elif k > l:
            seq_i_topk, _ = torch.topk(seq_i, l, dim=0)
            seq_i_topk_avg = torch.sum(seq_i_topk, dim=0) / l
        else:
            seq_i_topk, _ = torch.topk(seq_i, k, dim=0)
            seq_i_topk_avg = torch.sum(seq_i_topk, dim=0) / k
        b_seq_max_list.append(seq_i_topk_avg)

    return torch.cat(b_seq_max_list)


def comparing_conv_m(inputs, l1, l2, conv_layer: nn.Conv2d, mask_2d):
    batch_size = inputs.size(0)
    unit_d = conv_layer.out_channels // 2
    conv_out = conv_layer(inputs)

    a, b = torch.chunk(conv_out, 2, dim=1)
    gated_conv_out = a * F.sigmoid(b) * mask_2d[:, :unit_d, :, :]

    max_out_list = []
    for b_i in range(batch_size):
        b_conv_out = gated_conv_out[b_i, :, :l2[b_i], :l1[b_i]]
        max_out, _ = torch.max(b_conv_out.contiguous().view(unit_d, -1), dim=1)
        # print(b_conv_out.size())
        max_out_list.append(max_out.squeeze(1))
    max_out = torch.stack(max_out_list)

    return gated_conv_out, max_out


def text_conv1d(inputs, l1, conv_filter: nn.Linear, k_size, dropout=None, list_in=False,
                gate_way=True):
    """
    :param inputs: [T * B * D] 
    :param l1:  [B]
    :param conv_filter:  [k * D_in, D_out * 2]
    :param k_size:  
    :param dropout: 
    :param padding: 
    :param list_in: 
    :return: 
    """
    k = k_size
    batch_size = l1.size(0)
    d_in = inputs.size(2) if not list_in else inputs[0].size(1)
    unit_d = conv_filter.out_features // 2
    pad_n = (k - 1) // 2

    zeros_padding = Variable(inputs[0].data.new(pad_n, d_in).zero_())

    batch_list = []
    input_list = []
    for b_i in range(batch_size):
        masked_in = inputs[:l1[b_i], b_i, :] if not list_in else inputs[b_i]
        if gate_way:
            input_list.append(masked_in)

        b_inputs = torch.cat([zeros_padding, masked_in, zeros_padding], dim=0)
        for i in range(l1[b_i]):
            # print(b_inputs[i:i+k])
            batch_list.append(b_inputs[i:i+k].view(k * d_in))

    batch_in = torch.stack(batch_list, dim=0)
    a, b = torch.chunk(conv_filter(batch_in), 2, 1)
    out = a * F.sigmoid(b)

    out_list = []
    start = 0
    for b_i in range(batch_size):
        if gate_way:
            out_list.append(torch.cat((input_list[b_i], out[start:start + l1[b_i]]), dim=1))
        else:
            out_list.append(out[start:start + l1[b_i]])

        start = start + l1[b_i]

    # max_out_list = []
    # for b_i in range(batch_size):
    #     max_out, _ = torch.max(out_list[b_i], dim=0)
    #     max_out_list.append(max_out)
    # max_out = torch.cat(max_out_list, 0)
    #
    # print(out_list)

    return out_list

# Test something