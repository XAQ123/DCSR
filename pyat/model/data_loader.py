import json
import torch


class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self):
        self.batch_size = 32
        self.ptr = 0 # 遍历变量
        self.data = []

        data_file = 'data/train_set.json'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:# 获取数据
            self.data = json.load(i_f)
        with open(config_file) as i_f:# 获取数据配置信息
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], [] # ys记录做题记录结果 长度均为batch-size的长度，即32.
        for count in range(self.batch_size): # 获得batch-size（32）个train data中的记录
            log = self.data[self.ptr + count]
            knowledge_emb = [0.] * self.knowledge_dim # 生成这道题的知识点嵌入列表
            for knowledge_code in log['knowledge_code']: # 有的一道题目考察的知识点书不止一个，遍历这条做题记录中涉及到的每个知识点
                knowledge_emb[knowledge_code - 1] = 1.0 # 将这道题的知识点嵌入列表 相应位置 置1
            y = log['score']
            input_stu_ids.append(log['user_id'] - 1) # 将这条记录中的user-id添加到输入user-ids列表中
            input_exer_ids.append(log['exer_id'] - 1)# 将这条记录中的exer_id添加到输入exer_ids列表中
            input_knowedge_embs.append(knowledge_emb) # 将这道题的知识点考察列表添加的输入知识点考察列表中去
            ys.append(y) # 将这条记录的得分添加的ys中去

        self.ptr += self.batch_size # 遍历下一个batch-size个记录
        # 下面返回32个作答记录的user-id、exer-id、knowedge_emb、ys
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, d_type='validation'):
        self.ptr = 0
        self.data = []
        self.d_type = d_type

        if d_type == 'validation':
            data_file = 'data/val_set.json'
        else:
            data_file = 'data/test_set.json'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
            self.knowledge_dim = int(knowledge_n) # knowledge_dim是知识点的维度是123

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        logs = self.data[self.ptr]['logs'] # 获得第ptr个数据的logs 即问题的id、成绩和涉及的知识点。
        user_id = self.data[self.ptr]['user_id'] # 获得第ptr个数据的user—id
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], [] # 声明变量
        for log in logs: # 遍历第ptr个学生的的所有作答记录
            input_stu_ids.append(user_id - 1) # 将第ptr个数据的user-id添加到ids列表里来，user-id-1是让下标从0开始
            input_exer_ids.append(log['exer_id'] - 1)
            knowledge_emb = [0.] * self.knowledge_dim # 生成一个长度为knowledge_dim（123）的列表，且每个元素都是0.0
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0 # 这道题如果考察了这个知识点，那么知识点嵌入列表的相应知识点为1 -1是为了数组记录方便 kn_emb就是q矩阵中的那一行，就是为了得到那一行，因为公式里面要用到
            input_knowledge_embs.append(knowledge_emb) # 将第ptr个学生的每条作答记录中所考察的知识点列表添加到输入知识点列表中，input_knowledge_embs的长度就是这个学生作答的题目的数量
            y = log['score'] # y=这个学生在这道题上的作答情况为0或1
            ys.append(y) # 将这个学生每道题的作答情况都添加的ys里
        self.ptr += 1 # ptr+1，遍历下一个学生
        # 将这个学生的user-id列表、exer-id列表、知识点嵌入列表（包含了每道题所考察的知识点嵌入列表）、在每道题上的作答记录 返回。
        # 这四个列表的长度都等于数据中logs-num的长度，即这个学生的所有作答记录的数量。
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
