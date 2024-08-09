import torch
import torch.nn as nn


class Net(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n):
        self.knowledge_dim = knowledge_n # 获得知识点总数
        self.exer_n = exer_n # 获得试题总数
        self.emb_num = student_n # 获得学生总数
        self.stu_dim = self.knowledge_dim # 学生嵌入维度就是知识点的数量，以表示对每个知识点的掌握程度
        self.prednet_input_len = self.knowledge_dim # 预处理网络输入长度等于知识点数量
        self.prednet_len1, self.prednet_len2 = 512, 256  # 前两个全连接层的大小（输出维度）

        super(Net, self).__init__() # 调用父类的构造函数

        # network structure
        # Embedding解释：第一个参数表示词典的大小，在这里为学生的数量（也就是可训练矩阵的行数，用每一行这个向量来表示一个学生），即表示输入数据的类别数；
        # 第二个参数表示就是嵌入后的维度，即编码维度，就是用多少维来表示每一个学生，这个就是可训练矩阵的列数

        # 通过embedding得到这个学生对所有知识点的掌握程度
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim) # 默认0为填充数字
        # 通过embedding得到这个每个试题关于所有知识点的难度
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        # 通过embedding得到试题区分度e_discrimination
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        # 第一个全连接层
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1) # 线性层将123长度转换为512长度
        # 随机的将线性层中的一些神经元不发挥作用，防止模型过拟合
        self.drop_1 = nn.Dropout(p=0.5)
        # 第二个全连接层
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2) # 线性层将512长转换为256
        # 随机的将线性层中的一些神经元不发挥作用，防止模型过拟合
        self.drop_2 = nn.Dropout(p=0.5)
        # 第三个全连接层 256-》1  用最后那个一维的量sigmoid之后的结果来预测这个学生能不能答对这个题
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization 参数初始化。
        # 这个神经网络里有很多的参数，比如说embedding的可训练矩阵，线性层的权重和偏置。得给这些参数初始化一下，别让刚开始的参数值太差
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors # 这32个题考查的q矩阵
        :return: FloatTensor, the probabilities of answering correctly # 能够正确回答概率
        '''
        # before prednet
        '''
        student_emb|k_difficulty|e_discrimination 该函数先将输入数据由索引编号转换为onehot编码的矩阵，然后右乘一个权重矩阵完成
        输入的embedding   乘以kn_emb Q矩阵才不会出错。
        '''

        # stu_id是batch_size个（这个模型里32个）学生的ID，exer_id就是batch_size个（这个模型里32个）问题的ID
        # kn_emb是这batch_size个（这个模型里32个）问题对应的Q矩阵中的那一行，就是这个题考查了哪些知识点。

        # 将学生索引张量通过嵌入层嵌入，然后再非线性化，得到学生的嵌入，得到32个hs组成的矩阵，文章用每个hs表示每个学生对123个知识点的掌握程度
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        # 将32个试题嵌入，再通过非线性化，得到32个题的关于知识点的难度向量组成的矩阵。
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        # 将试题ID嵌入到1维空间中，文章用这个嵌入后的值表示这个题的区分度。
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10 # 将得到的试题区分度嵌入*10，改变区分度的范围 从0-1到0-10
        # prednet 开始用诊断公式诊断，下一行文章最精彩部分
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb # 学生的掌握向量就是学生的嵌入stu_emb kn_emb就是q矩阵中的那一行
        # 一个全连接层一个激活函数一个dropout防止过拟合 123-》512 升维 这个过程有很多神经元可以让模型的参数往更准确的值调整
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x))) # 第一个全连接层
        # 一个全连接层一个激活函数一个dropout防止过拟合 512-》256 降维 这个过程有很多神经元可以让模型的参数往更准确的值调整
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x))) # 第二个全连接层
        # 先全连接层后激活 256-》1 sigmoid后的值就是来预测这个学生答对这道题的概率 得到的输出是32个答对相应题目的概率
        output = torch.sigmoid(self.prednet_full3(input_x)) # 第三个全连接层
        return output

    def apply_clipper(self): # 把线性层里面的参数中负的都变成正的 一个全连接层优化的部分。
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id): # 得到学生的知识状态
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id): # 得到试题参数 难度和区分度
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data

# 全连接层优化
class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a) # 把原来是负数的加上relu之后的正数

