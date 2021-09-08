import   pandas  as  pd

import   jieba
from gensim import models

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import  gensim

import  numpy  as np

import  os
import  torch
import torch.nn as nn
import torch.functional as F


class   RNNConfig():
    def __init__(self,vocabSize,outputSize=2,batchSize=50,embedDimention=300,
                 hiddenSize=64,hiddenLayer=3,dropKeep=0.1,bidirectional=True,
                 lr=0.001,cuda=False,saveDir="./data/snap/",
                 logInteval=5,epochs= 3 ,evalInteval=-1,preTrain=True,embdingVect=None
                 ):

        self.vocabSize= vocabSize                    # 总词数 多少个
        self.batchSize = batchSize                   #一次性传入多少数据
        self.embedDimention= embedDimention            #词向量维度
        self.hiddenSize = hiddenSize                      # lstm隐藏层大小
        self.hiddenLayer= hiddenLayer                  # lstm层数
        self.dropKeep= dropKeep                       # 随机失活
        self.bidirectional= bidirectional             # 是否双向
        self.outputSize=outputSize                    #输出大小
        self.lr= lr                                  #学习率
        self.cuda= cuda                               #是否用GPU
        self.saveDir = saveDir                        #保存快照的位置
        self.logInteval =logInteval                    #隔多少步 打一个log
        self.epochs= epochs                           # 循环多少次
        self.evalInteval = evalInteval                #隔多少步 评估一下保存快照
        self.preTrain= preTrain
        self.embdingVect = embdingVect


class TextRNN(nn.Module):
    def __init__(self, config):
        super(TextRNN, self).__init__()
        self.config = config

        if  config.preTrain:
            # self.embeddings = nn.Embedding(self.config.vocabSize, self.config.embedDimention)
            #
            # self.embeddings.weight.data.copy_(config.embdingVect)
            self.embeddings = nn.Embedding.from_pretrained(config.embdingVect,freeze=False)

            print("pr train")
        else:

            # Embedding 层， 随机初始化

            self.embeddings = nn.Embedding(self.config.vocabSize, self.config.embedDimention)

        # LSTM 层
        '''
        input_size:输入特征的数目
        hidden_size:隐层的特征数目
        num_layers：这个是模型集成的LSTM的个数 记住这里是模型中有多少个LSTM摞起来 一般默认就1个
        #batch_first: 输入数据的size为[batch_size, time_step, input_size]还是[time_step, batch_size, input_size]
       '''
        self.lstm = nn.LSTM(input_size=self.config.embedDimention,
                            hidden_size=self.config.hiddenSize,
                            num_layers=self.config.hiddenLayer,
                            dropout=self.config.dropKeep,
                            bidirectional=self.config.bidirectional,
                            batch_first=True
                            )
    

        # dropout
        self.dropout = nn.Dropout(self.config.dropKeep)

        outSize= self.config.hiddenSize   * ( 2 if  self.config.bidirectional  else 1 ) #*self.config.hiddenLayer

        print("outSize=",outSize)
        # 全连接层
        self.fc = nn.Linear(  # 就是 hn、cn 的输出然后去掉 batch_size
            outSize,
            self.config.outputSize
        )

        # softmax 层
        self.softmax = nn.Softmax(dim=1)

        # for  param  in  self.parameters():
        #      print("parm==",param)

        self.optimizer= torch.optim.Adam(self.parameters(),config.lr)
        self.lossFunc = nn.CrossEntropyLoss()








    def RunModel(self,x):
        # x.shape = (max_sen_len, batch_size)

        #x = torch.LongTensor(x)
        print("x:",x.size(),x[0])

        embedded_sent = self.embeddings(x)  # (max_sen_len = 30, batch_size=64, embed_size=300)

        embedded_sent = self.dropout(embedded_sent)

        print("embedded_sent==",embedded_sent.size())



        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded_sent,None)

        # dropout
        final_feature_map = self.dropout(h_n)  # (num_layers * num_directions, batch_size, hidden_size)

        print("final_feature_map:",final_feature_map.size())



        final_feature_map = torch.cat((final_feature_map[-1, :, :] ,final_feature_map[-2, :, :]), dim=1)

        print("final_feature_map22:", final_feature_map.size())
        # 全连接
        final_out = self.fc(final_feature_map)
        #final_out = self.softmax(final_out)


        return final_out # 返回 softmax 的结果



    def forward(self, x):
        return  self.RunModel(x)


    def  Refrush(self,predictY,targetY):
        self.optimizer.zero_grad()
        loss= self.lossFunc(predictY,targetY)
        loss.backward()
        self.optimizer.step()
        print("loss:",loss.data.item())

        return  loss


    def ShowRate(self,prdictY,targetY):
        result = torch.argmax(prdictY,dim=1)
        print("rate==",prdictY[0:5],result[0:5],targetY[0:5])
        corrects = (result == targetY).sum().item()

        accuracy = corrects  / self.config.batchSize
        print ("correct:",corrects,"acc:",accuracy)

    def SaveMode(self,saveDir,step):
        if not  os.path.exists(saveDir):
            os.mkdir(saveDir)
        savePath = "{}Steps_{}.pt".format(saveDir,step)
        torch.save(self.state_dict(),savePath)

    def RunTrain(self,trainIter,evalIter):

         step =0
         bestAcc = 0
         self.train()
         for   epoch  in  range (1,self.config.epochs+1):
             for batch in  trainIter:
                 feature, target = batch.text, batch.label


                 if self.config.cuda:
                     feature,target = feature.cuda(),target.cuda()
                 predictY = self.RunModel(feature)
                 print("predict Y:",predictY.size(),target.size())
                 loss =self.Refrush(predictY,target)

                 if loss.data.item() < 0.0001:
                     break

                 step += 1
                 if step % self.config.logInteval ==0:
                     self.ShowRate(predictY, target)

                 if  self.config.evalInteval >0  and step % self.config.evalInteval ==0  :
                     devAcc= self.Eval(evalIter)
                     if devAcc > bestAcc:
                         bestAcc = devAcc
                         #self.SaveMode(self.config.saveDir,step)
                     self.train()




    def  Eval(self,dataIter):
        self.eval()
        avgLoss =0.0
        corrects=0.0
        accuracy=0.0
        for batch in dataIter:
            feature, target = batch.text, batch.label
            #feature.data.t_()

            if self.config.cuda:
                feature, target = feature.cuda(), target.cuda()

            predictY = self.RunModel(feature)

            loss = F.cross_entropy(predictY,target)
            avgLoss += loss.item()
            #result = torch.max(predictY, 1)[1]


            result = torch.argmax(predictY, dim=1)
            print("rate==", predictY[0:5], result[0:5], target[0:5])
            correct = (result == target).sum().item()
            acc= correct / self.config.batchSize
            accuracy += acc
            print("correct:", correct, "acc:", acc)

        size =len(dataIter)
        avgLoss /= size

        accuracy =  accuracy /size


        print("eval loss:{} acc:{}".format(avgLoss,accuracy))
        return  accuracy

