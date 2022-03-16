import torch
import torch.nn as nn


def change_weights(student_model, teacher_model, mode):
    if mode not in ['zero', 'uniform']:
        return 

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
