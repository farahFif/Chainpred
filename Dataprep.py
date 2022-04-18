import os
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import numpy as np


class Data_prep():
    def __init__(self,path):
        self.path = path
        self.train , self.valid , self.test = self.get_data(self.path)
        self.model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')

        tags = np.unique(self.train['TAG'].values)
        uniqtags = []
        for i in tags:
            for g in i:
                uniqtags.append(g)
        uniqtags = np.unique(uniqtags)
        print(uniqtags)
        self.tag_2_label = {}
        self.label_2_tag = {}
        cpt = 0
        for k, id in enumerate(uniqtags):
            self.tag_2_label[id] = cpt
            self.label_2_tag[cpt] = id
            cpt += 1
        self.tag_2_label['SOS'] = 100
        self.tag_2_label['EOS'] = 200
        self.label_2_tag[100] = 'SOS'
        self.label_2_tag[200] = 'EOS'
        print(self.tag_2_label)
        print(self.label_2_tag)

    def load_data(self,hops=3):
        train, valid,test = pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
        for h in range(1,hops+1):
            for i in os.listdir(os.path.join(self.path,str(h)+'hop/')):
                if str(i).startswith('.'): continue
                tmp_path = os.path.join(self.path,str(h)+'hop/')
                df = pd.read_csv(os.path.join(tmp_path, i),sep ='\t',names=['QA','ANS','TAG'],header=None)
                if i == 'train.txt':
                    train = pd.concat([train,df],axis=0)
                if i == 'valid.txt':
                    valid = pd.concat([valid,df],axis=0)
                if i == 'test.txt':
                    test = pd.concat([test,df],axis=0)
                print('hope',h,'file ',i, 'len ',len(df))

        train.to_csv('all_train.csv', index=False)
        valid.to_csv('all_valid.csv', index=False)
        test.to_csv('all_test.csv', index=False)
        return train , valid , test

    def get_data(self,path):
        if os.path.exists('data/all_train.csv'):
            train = pd.read_csv('data/all_train.csv').drop('Unnamed: 0', axis=1)
            print('traiiin ', len(train))

            valid = pd.read_csv('data/all_valid.csv').drop('Unnamed: 0', axis=1)
            test = pd.read_csv('data/all_test.csv').drop('Unnamed: 0', axis=1)
            return train ,valid,test
        else :
            return self.load_data()


    def fix_data(self):
        '''
            turn tags into a list as we may get more than one tag
            remove brackets from question
        :return:
        '''
        pattern = r'[\([{})\]]'
        self.train.dropna(inplace = True)
        self.test.dropna(inplace = True)
        self.valid.dropna(inplace = True)

        self.train['QA'] = self.train['QA'].apply(lambda x : re.sub(pattern, '', x))
        self.valid['QA'] = self.valid['QA'].apply(lambda x : re.sub(pattern, '', x))
        self.test['QA'] = self.test['QA'].apply(lambda x : re.sub(pattern, '', x))

        ### split tags
        self.train['TAG'] = self.train['TAG'].apply(lambda x: x.split('|'))
        self.valid['TAG'] = self.valid['TAG'].apply(lambda x:x.split('|'))
        self.test['TAG'] = self.test['TAG'].apply(lambda x: x.split('|'))

    def prep_for_train(self):
        self.train_embeddings = self.model.encode(self.train['QA'].values)
        self.y_train = []
        for label in self.train['TAG'].values:
            tmp = [self.tag_2_label['SOS']]
            for k in label:
                tmp.append(self.tag_2_label[k])
            tmp.append(self.tag_2_label['EOS'])
            self.y_train.append(tmp)

    def prep_for_valid(self):
        self.valid_embeddings = self.model.encode(self.valid['QA'].values)
        self.y_valid = []
        for label in self.valid['TAG'].values:
            tmp = [self.tag_2_label['SOS']]
            for k in label:
                tmp.append(self.tag_2_label[k])
            tmp.append(self.tag_2_label['EOS'])
            self.y_valid.append(tmp)

    def prep_for_test(self):
        self.test_embeddings = self.model.encode(self.test['QA'].values)
        self.y_test = []
        for label in self.test['TAG'].values:
            tmp = [self.tag_2_label['SOS']]
            for k in label:
                tmp.append(self.tag_2_label[k])
            tmp.append(self.tag_2_label['EOS'])
            self.y_test.append(tmp)






