import   pandas  as  pd

import   jieba
from gensim import models

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import  gensim

import  numpy  as np

import  os
import  torch
import torch.nn.functional as F
from torch import nn


'''
后面训练 和测试的 数据格式
'''
class  Batch(object):
    def __init__(self,label,text):
        self.text =torch.LongTensor(text)
        self.label=torch.LongTensor([int(x) for x in  label])



class  DataSet(object):

    def __init__(self):
        self.UNK, self.PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

    '''
    # 将 csv 文件 变成 [  [ 文档1] ,[ 文档2]]
    '''
    def  ReadCSV(self,path):
         csvFile=  pd.read_csv(path,usecols=[1,2],encoding="utf-8")

         labelList  = csvFile.values[:,0].tolist()
         valueList = csvFile.values[:,1].tolist()

         print( labelList[0],valueList[0])

         print(len(valueList))


         return   labelList,valueList


    '''
    将tsv 文件变成  [  ["我","爱" ，"小喵咪"]  ,["我","吃" ,"西瓜" ] ... ]
    '''
    def ReadTSV(self,path):

        label=[]
        text=[]
        with  open(path,"r",encoding="utf-8") as fhandle:
           line=  fhandle.readline()

           while line :
               line = fhandle.readline()
               lines= line.split("\t")

               if len(lines) != 3:
                   continue

               label.append(lines[1])

               tmpLine= [word  for  word in jieba.cut(lines[2])]


               #print(lines)
               #print("line=",len(lines),lines[1],tmpLine)

               text.append(tmpLine)

        return  label,text


    '''
    将数据[[],[]]切割成Batch 块 [ batch1,batch2]
    '''
    def  SpliteData(self,text,label,batchSize):
         allLen = len(text)
         step = int(allLen/batchSize)

         spliteText=[]

         for i in  range(0,step):
             spliteText.append(Batch(label[i:i+batchSize] ,text[i:i+batchSize]))


         return  spliteText




    '''
    将解析tsv文件
    数据变成等长的 list [ [1, 2, 5],  [1, 2, 4]  ...]
    切割成[ batch1,batch2]
    '''
    def BuidBatch(self,batchSize,vocabDict,sentenSize=32,
                  testPath="./data/test_mini.tsv",valPath="./data/dev_mini.tsv",trainPath="./data/train_mini.tsv"):

        testLabel,testText = self.ReadTSV(testPath)
        valLabel,valText  = self.ReadTSV(valPath)
        trainLabel,trainText = self.ReadTSV(trainPath)

        testText = self.Word2Vect(testText,vocabDict,sentenSize)
        valText = self.Word2Vect(valText,vocabDict,sentenSize)
        trainText = self.Word2Vect(trainText, vocabDict, sentenSize)



        testIter = self.SpliteData(label=testLabel,text= testText,batchSize=batchSize)

        valIter= self.SpliteData(label=valLabel,text= valText,batchSize=batchSize)
        trainIter = self.SpliteData(label=trainLabel, text=trainText, batchSize=batchSize)

        return  testIter,valIter,trainIter






    '''
    splitTexts =[["xx","xxx"],["xx","xxx"]]

    vocabDict=[('空间', index1), ('很', index2),]
    '''
    def  BuildVocabDict(self,splitTexts,minFreq=1):
          vocabDict={}
          for   setences in   splitTexts:
              #print("setence==",setences)
              for word in  setences:

                  vocabDict[word] = vocabDict.get(word,0)+1

          vocabList = sorted([_ for _ in vocabDict.items() if _[1] >= minFreq], key=lambda x: x[1], reverse=True)

          print("vocabList==",vocabList)
          vocabDict = {word_count[0]: idx for idx, word_count in enumerate(vocabList)}

          vocabDict.update({self.UNK: len(vocabDict),self.PAD: len(vocabDict) + 1})

          print("vocabDict size",len(vocabDict))

          return   vocabDict






    #将词向量存到硬盘
    def  Word2Vect(self, docments,vocabDict,embSize=300,prePath="./data/"):
        model = gensim.models.Word2Vec(docments, sg=1, size=embSize, window=5, min_count=1, negative=3, sample=0.001, hs=1,
                                       workers=4)

        #model.wv.save_word2vec_format("word300.txt",binary=False)

        wordEmbding=[]
        found_numb = 0
        not_found_numb = 0
        counts = len(docments)
        for   k,v in  vocabDict.items():

            #print("k,",k )
            if(model.wv.__contains__(k)):
                tmpVect = model.wv.get_vector(k)
                #print("model==",tmpVect)
                wordEmbding.append(tmpVect)
                found_numb += 1
            else:
                embedding = np.random.uniform(0, 1, embSize)
                wordEmbding.append(embedding)
                not_found_numb += 1
                # print("not contain key==",k)

        print(f"word_vec train counts:{counts}, found:{found_numb}, not found:{not_found_numb}")

        # fileName = "{}word{}.npz".format(prePath,embSize)
        # if not os.path.exists(fileName):
        #     fd = open(fileName,"w",encoding="utf-8")
        #     fd.close()

        # np.savez(fileName,vocabDict=vocabDict,wordEmbding=wordEmbding)


        return  wordEmbding

    #加载词向量
    def  LoadWordEmbding(self,path="./data/word300.npz"):

        savNpz=  np.load(path,allow_pickle=True)
        embedding_pretrained =  savNpz["wordEmbding"].astype('float32')
        vocabDict = savNpz["vocabDict"].item()

        print("embding===",len(embedding_pretrained))
        print("embding dict=",len(vocabDict))

        return  vocabDict, embedding_pretrained

    #建立词向量
    def  BuildWordVect(self,embSize):

        if not os.path.exists("./data/word300.npz"):
            print("word_vec is not exist, beginning train word_vec...")

            label, text = self.ReadCSV("./data/ch_auto.csv")

            splitTexts = [[word for word in jieba.cut(setences)] for setences in text]

            # print(splitTexts[0])

            vocabDict = self.BuildVocabDict(splitTexts)

            self.Word2Vect(splitTexts, vocabDict, embSize=embSize)
            
            print("word_vec train end!")


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


#一次性传入的文档数量
BATHSIZE = 50
#每个句子的统一长度
SentenceLength = 32
#词向量维度
EmbdingDemition=300


if __name__ == '__main__':

    dataSet = DataSet()

    ## 训练词向量
    dataSet.BuildWordVect(EmbdingDemition)


    ##加载词向量
    vocabDict,wordVect= dataSet.LoadWordEmbding()

    ## 准备训练数据
    trainIter, valIter, testIter =dataSet.BuidBatch(BATHSIZE,vocabDict,sentenSize=SentenceLength)


    ## 配置文件
    config = RNNConfig(len(vocabDict),embedDimention=EmbdingDemition,batchSize=BATHSIZE,
                       preTrain=True,embdingVect= torch.tensor(wordVect)
                       )

    ## 初始化 RNN
    myRNN= TextRNN(config)

    ## 开始训练
    myRNN.RunTrain(trainIter,valIter)

    ## 开始测试
    myRNN.Eval(testIter)

