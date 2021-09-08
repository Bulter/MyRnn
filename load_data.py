import   pandas  as  pd

import   jieba
from gensim import models

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import  gensim

import  numpy  as np

import  os
import  torch


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
                  testPath="./data/test.tsv",valPath="./data/dev.tsv",trainPath="./data/train.tsv"):

        testLabel,testText = self.ReadTSV(testPath)
        valLabel,valText  = self.ReadTSV(valPath)
        trainLabel,trainText = self.ReadTSV(trainPath)

        testText = self.Doc2Embedding(testText,vocabDict,sentenSize)
        valText = self.Doc2Embedding(valText,vocabDict,sentenSize)
        trainText = self.Doc2Embedding(trainText, vocabDict, sentenSize)



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
        for   k,v in  vocabDict.items():

            #print("k,",k )
            if(model.wv.__contains__(k)):
                tmpVect = model.wv.get_vector(k)
                #print("model==",tmpVect)
                wordEmbding.append(tmpVect)
            else:
                embedding = np.random.uniform(0, 1, embSize)
                wordEmbding.append(embedding)
                print("not contain key==",k)



        fileName = "{}word{}.npz".format(prePath,embSize)
        if not os.path.exists(fileName):
            fd = open(fileName,"w",encoding="utf-8")
            fd.close()

        np.savez(fileName,vocabDict=vocabDict,wordEmbding=wordEmbding)


        return  model

    #加载词向量
    def  LoadWordEmbding(self,path="./data/word300.npz"):

        savNpz=  np.load(path,allow_pickle=True)
        embedding_pretrained =  savNpz["wordEmbding"].astype('float32')
        vocabDict = savNpz["vocabDict"].item()

        print("embding===",len(embedding_pretrained))
        print("embding dict=",vocabDict)

        return  vocabDict, embedding_pretrained

    #建立词向量
    def  BuildWordVect(self,embSize):

        if not os.path.exists("./data/word300.npz"):
            label, text = self.ReadCSV("./data/ch_auto.csv")

            splitTexts = [[word for word in jieba.cut(setences)] for setences in text]

            print(splitTexts[0])

            vocabDict = self.BuildVocabDict(splitTexts)

            self.Word2Vect(splitTexts, vocabDict, embSize=embSize)

