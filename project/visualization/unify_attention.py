import torch
import torch.nn.functional as F


def average_heads_attentions_CLS(batch: torch.FloatTensor) -> torch.FloatTensor:
    r"""
    It calculate the average of attention that each CLS tokens have.

    :param batch: shape(seq_len, seq_len)
    :return: averaged
    """
    head_num, seq_len = len(batch), len(batch[0])
    sum_heads_atten = torch.zeros(seq_len)
    for head in batch:
        sum_heads_atten += torch.tensor([atten for atten in head[0]])  # add attention that each CLS tokens have
    return sum_heads_atten / head_num


def average_attentions(attentions: tuple[torch.FloatTensor], temp: float = 0.0005) -> torch.Tensor:
    r"""
    Calculate the average of all layer's attention.(2021/3/8)
    check the latest transformers document if it don't work well.
    (https://huggingface.co/transformers/main_classes/output.html#basemodeloutput)
    :param attentions: Tuple of torch.FloatTensor(one for each layer) of shape(batch_size, num_heads, seq_len, seq_len)
    :param temp: temperature
    :return: averaged attention. shape(batch_size, seq_len)
    """
    layer_num, batch_size, head_num, seq_len = len(attentions), len(attentions[0]), len(attentions[0][0]), len(attentions[0][0][0])
    sum_layers_atten = torch.zeros(batch_size, seq_len)
    for layer in attentions:  # len(attentions) = layer_num
        for i, batch in enumerate(layer):  # len(layer) = batch_size
            ave_heads_atten = average_heads_attentions_CLS(batch)
            sum_layers_atten[i] += ave_heads_atten
    return softmax_with_temp(sum_layers_atten / layer_num, dim=0, temp=temp)


def average_last_layer_attentions(attentions: tuple[torch.FloatTensor], temp: float = 0.0005) -> torch.Tensor:
    r"""
    It calculate the average of last layer's attention.(2021/3/8)
    check the latest transformers document if it don't work well.
    (https://huggingface.co/transformers/main_classes/output.html#basemodeloutput)
    :param attentions: Tuple of torch.FloatTensor(one for each layer) of shape(batch_size, num_heads, seq_len, seq_len)
    :param temp: temperature
    :return: averaged attention. shape(batch_size, seq_len)
    """
    last_layer = attentions[-1]
    batch_size, head_num, seq_len = len(last_layer), len(last_layer[0]), len(last_layer[0][0])
    ave_last_layer_atten = torch.zeros(batch_size, seq_len)
    for i, batch in enumerate(last_layer):
        ave_heads_atten = average_heads_attentions_CLS(batch)
        ave_last_layer_atten[i] = softmax_with_temp(ave_heads_atten, dim=0, temp=temp)
    return ave_last_layer_atten


def softmax_with_temp(inp: torch.tensor, dim: int, temp: float):
    r"""
    to adjust contrast of attention

    :param inp: a row of attention
    :param dim: a dimension along with which softmax will be computed
    :param temp: temperature
    :return:
    """
    return F.softmax(inp / temp, dim=dim)

