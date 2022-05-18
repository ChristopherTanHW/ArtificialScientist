from distutils import command
import os

args_dict = {
'env':'rock_paper_scissors-v0',
'model':'paper_txt2pi',
'demb':'10',
'drnn_small': '10',
'drnn':'100',
'drep':'300',
'num_actors':'20',
'batch_size':'24',
'learning_rate':'0.0007',
'total_frames':'50000000',
'height':'10',
'width':'10',
'mode': 'train_writer',
'xpid': 'RPSdefault_butsize10'}
# 'resume': 'checkpoints/groups_simple_stationary:paper_txt2pi:yeswiki:default/model.tar'}

#groups env
# args_dict = {
# 'env': 'groups_simple_stationary-v0',
# 'model':'paper_txt2pi',
# # 'demb':'30',
# # 'drnn_small': '10',
# # 'drnn':'100',
# # 'drep':'400',
# # 'num_actors':'20',
# # 'batch_size':'24',
# # 'learning_rate':'0.0007',
# # 'total_frames':'300000',
# # 'height':'6',
# # 'width':'6',
# 'mode': 'test',
# # 'xpid': 'groups_simple_stationary_resume_constant_game_0lie', #for training
# 'xpid': 'groups_simple_stationary:paper_txt2pi:yeswiki:default', #for testing
# 'resume': 'checkpoints/groups_simple_stationary:paper_txt2pi:yeswiki:default/model.tar'}

#this particular experiment looks at learning rate
for i in range(1,2):
    # args_dict['learning_rate'] = str(0.0007)
    command_string = "python run_exp.py "

    for k, v in args_dict.items():
        command_string += '--' + k + " " + v + " "

    os.system(command_string)


'''command_string = "python run_exp.py "
for k, v in args_dict.items():
    command_string += '--' + k + " " + v + " "
os.system(command_string)'''