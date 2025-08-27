import torch
import torchac


def calc_tensor_bits(values):
    # print(values)
    min_v, max_v = values.min(), values.max()
    values_norm = values - min_v
    symbols = torch.arange(min_v, max_v + 1)

    pmf = torch.softmax(torch.abs(torch.ones([1, symbols.shape[0]])*0.5), -1)
    cdf = pmf.cumsum(-1)
    zeros = torch.zeros([1, pmf.shape[0]])
    cdf = torch.cat([zeros, cdf], dim=-1)

    cdf = cdf.reshape(1, 1, -1).repeat(values_norm.shape[0], values_norm.shape[1], 1)  # 每个要编码的符号都给一行cdf
    string = torchac.encode_float_cdf(cdf, values_norm)

    values_norm_d = torchac.decode_float_cdf(cdf, string)
    values_d = values_norm_d + min_v
    # print(values_d)
    return len(string)*8




