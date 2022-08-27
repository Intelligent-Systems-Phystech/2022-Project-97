import torch.nn as nn 
import torch

from consts import device
from pipeline import make_student_model

def simple_baseline_change_weights(teacher_model, mode):
    student_model = make_student_model()
    if mode not in ['zero', 'uniform']:
        raise ValueError('bad mode')

    student_model.stack[0][0].weight = nn.Parameter(teacher_model.stack[0][0].weight)
    student_model.stack[0][0].bias = nn.Parameter(teacher_model.stack[0][0].bias)

    if mode == 'zero':
        student_model.stack[1][0].weight = nn.Parameter(torch.cat((teacher_model.stack[1][0].weight,
                                                                   torch.zeros(64, 128).to(device))))
        student_model.stack[1][0].bias = nn.Parameter(torch.cat((teacher_model.stack[1][0].bias,
                                                                 torch.zeros(64).to(device))))
        
        student_model.stack[2][0].weight = nn.Parameter(torch.cat((teacher_model.stack[2][0].weight,
                                                                   torch.zeros(32, 64).to(device)), dim=1))
    elif mode == 'uniform':
        student_model.stack[1][0].weight = nn.Parameter(torch.cat((teacher_model.stack[1][0].weight,
                                                                   student_model.stack[1][0].weight[64:])))
        student_model.stack[1][0].bias = nn.Parameter(torch.cat((teacher_model.stack[1][0].bias,
                                                                 student_model.stack[1][0].bias[64:])))
        
        student_model.stack[2][0].weight = nn.Parameter(torch.cat((teacher_model.stack[2][0].weight,
                                                                   student_model.stack[2][0].weight[:, 64:]), dim=1))

    student_model.stack[2][0].bias = nn.Parameter(teacher_model.stack[2][0].bias)

    student_model.stack[3][0].weight = nn.Parameter(teacher_model.stack[3][0].weight)
    student_model.stack[3][0].bias = nn.Parameter(teacher_model.stack[3][0].bias)
    return student_model