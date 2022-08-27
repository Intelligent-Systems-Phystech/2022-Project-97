import torch 
data_path = './data/'
num_workers = 2
batch_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using {} device'.format(device))

colab_path = '/content/drive/MyDrive/models/'
local_path = './models/'
use_colab = False 

num_repeats = 2


full_teacher_training_epochs = 50
full_student_training_epochs = 50
full_teacher_learning_rate = 1e-4 
full_student_learning_rate = 1e-4 

teacher_5_training_epochs = 30
student_5_training_epochs = 30
teacher_5_learning_rate = 1e-4
student_5_learning_rate = 1e-4 

noise_eps = [i/100 for i in range(10)]

fsgm_eps = [i/500 for i in range(10)]