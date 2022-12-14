try:
    import pandas as pd
    import numpy as np
    import datetime
    from tqdm import tqdm
    import re
    import csv
    import os

    import torch
    from transformers import BertTokenizer
    from transformers import XLMRobertaTokenizer, XLMRobertaModel
    from transformers import RobertaTokenizer, RobertaModel
    from torch import nn
    from transformers import BertModel
    from torch.optim import Adam
    import time
except:
    print('Environment have not been setup! Please, install pytorch, tqdm, regex, datetime and transformers packages.')


if not os.path.isdir('CP'):
    os.mkdir('CP')
    print('Creating CP directory...')
else:
    add_str = str(time.localtime().tm_year) + str(time.localtime().tm_mon) + str(time.localtime().tm_mday) + str(time.localtime().tm_hour) + str(time.localtime().tm_min)
    os.rename('CP','CP_'+add_str)
    print('Renamed previous {} folder with {}'.format('CP','CP_'+add_str))
    os.mkdir('CP')
    print('Creating CP directory...')

# def CleanData(input_str, sel_pre):
#     if '0' in sel_pre:
#         input_str = re.sub('\)','',re.sub('\(', '', str(input_str)).strip()).strip()
#     if '1' in sel_pre:
#         input_str = re.sub('\]','',re.sub('\[', '', input_str).strip()).strip()
#     if '2' in sel_pre:
#         input_str = re.sub(re.compile(','), '', input_str).strip()
#     if '3' in sel_pre:
#         input_str = re.sub(re.compile('\\.'), '', input_str).strip()
#     if '4' in sel_pre:
#         input_str = re.sub(re.compile('\''), '', str(input_str)).strip()
#     return input_str

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer,is_train=True):
        input_data = data['input'].apply(lambda x: str(x))
#         print('Applying pre-processing on entire dataset')
#         input_data = CleanData(input_data, sel_pre)
        if is_train:
            un_label = data['target'].unique()
            print('Saving target encoding (hot encoding for label) format in encode_target.npy file')
            np.save('encode_target.npy',un_label)
            print('Using {} number of unique classes for the BERT model training'.format(len(un_label)))
        else:
            un_label = np.load('encode_target.npy',allow_pickle=True)
        self.labels = [np.where(un_label==label)[0][0] for label in data['target']]
        self.texts = [tokenizer(str(text), 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in input_data]
        self.tx_orig = [str(i) for i in input_data]

    def classes(self):
        return self.labels
    def __len__(self):
        return len(self.labels)
    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])
    def get_batch_ori_text(self, idx):
        return self.tx_orig[idx]
    def get_batch_texts(self, idx):
        return self.texts[idx]
    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        batch_or_tx = self.get_batch_ori_text(idx)

        return batch_texts, batch_y, batch_or_tx

class BertClassifier(nn.Module):
    def __init__(self, dropout,which_bert,num_class):
        super(BertClassifier, self).__init__()
        if which_bert=='bert':
            self.bert = BertModel.from_pretrained('bert-base-cased')
        elif which_bert=='ml-bert':
            self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        elif which_bert=='roberta':
            self.bert = RobertaModel.from_pretrained('roberta-base')
        else:
            self.bert = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_class)
        self.sigm = nn.Sigmoid()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.sigm(linear_output)

        return final_layer

def train(model, train_data, val_data, tokenizer, batch_size, learning_rate, epochs):
    print('Preparing data loader...')
    train, val = Dataset(train_data, tokenizer), Dataset(val_data, tokenizer,is_train=False)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    
    v_acc=[]
    t_acc=[]
    e_list=[]
    for epoch_num in range(epochs):
        e_list.append(epoch_num)
        print('Model Training at Epoch: ' + str(epoch_num))
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label, tr_or_tx in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0
        torch.save(model.state_dict(), 'CP/' + str(epoch_num) + '_model.pth')
        print('Model saved! Calculating validation accuracy...')
        t_acc.append(total_acc_train/len(train_data))
        with torch.no_grad():

            for val_input, val_label, val_or_tx in tqdm(val_dataloader):

                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        v_acc.append(total_acc_val/len(val_data))
        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
            | Train Accuracy: {total_acc_train / len(train_data): .3f} \
            | Val Loss: {total_loss_val / len(val_data): .3f} \
            | Val Accuracy: {total_acc_val / len(val_data): .3f}')
        res_df = pd.DataFrame({'Epoch':e_list, 'Train Accuracy':t_acc, 'Val Accuracy':v_acc})
        res_df.to_csv('log.csv')


def BERT_Model(train_data,valid_data,which_bert='bert',dropout=0.5,EPOCHS=20,LR=0.000001,batch_size=8,pretrain_model=''):
    print('Replacing all nan with empty value')
    train_data = train_data.replace(np.nan,'')
    valid_data = valid_data.replace(np.nan,'')
#     col = train_data.columns
#     print('Obtaining input data information')
#     k=0
#     print('Select the column index for input of the BERT model.')
#     for i in col:
#         print('{}-{}'.format(k,i))
#         k+=1

#     response = input('Your response (separated by comma): ')
#     sel_col_ind = response.split(',')

#     train_new_data = pd.DataFrame(columns=['input','target'])
#     valid_new_data = pd.DataFrame(columns=['input','target'])
#     for i in sel_col_ind:
#         if i==sel_col_ind[0]:
#             train_new_data['input'] = train_data[col[int(i)]]
#             valid_new_data['input'] = valid_data[col[int(i)]]
#         else:
#             train_new_data['input'] = train_new_data['input'] + ' ' + train_data[col[int(i)]]
#             valid_new_data['input'] = valid_new_data['input'] + ' ' + valid_data[col[int(i)]]

#     print('Obtaining output data information')
#     k=0
#     print('Select the column index for target of the BERT model.')
#     for i in col:
#         print('{}-{}'.format(k,i))
#         k+=1

#     response = input('Your response (separated by comma): ')
#     sel_col_ind = response.split(',')

#     for i in sel_col_ind:
#         if i==sel_col_ind[0]:
#             train_new_data['target'] = train_data[col[int(i)]].str.lower()
#             valid_new_data['target'] = valid_data[col[int(i)]].str.lower()
#         else:
#             train_new_data['target'] = train_new_data['target'] + '_' + train_data[col[int(i)]].str.lower()
#             valid_new_data['target'] = valid_new_data['target'] + '_' + valid_data[col[int(i)]].str.lower()

#     n_old=len(train_new_data)
#     train_new_data = train_new_data[train_new_data['input']!='']
#     train_new_data = train_new_data[train_new_data['target']!='_']
#     n_after=len(train_new_data)
#     print('Removed {} items having empty and nan value from {} training items'.format(n_old-n_after,n_after))
    
#     n_old=len(valid_new_data)
#     valid_new_data = valid_new_data[valid_new_data['input']!='']
#     valid_new_data = valid_new_data[valid_new_data['target']!='_']
#     n_after=len(valid_new_data)
#     print('Removed {} items having empty and nan value from {} validation items'.format(n_old-n_after,n_after))

#     pre_task = ['\( and \) removal', '\[ and \] removal','Comma removal', 'period removal', '\' removal']
#     k=0
#     print('Select the index for pre-processing of the input data.')
#     for i in pre_task:
#         print('{}-{}'.format(k,i))
#         k+=1

#     response = input('Your response (separated by comma): ')
#     sel_pre = response.split(',')
    
#     which_bert = input('Press 1 for English BERT and 2 for multilingual BERT: ')
#     print('1-English BERT')
#     print('2-Multilingual BERT')
#     print('3-English RoBERTa')
#     print('4-Multilingual RoBERTa')
#     which_bert = input('Select which BERT model you require:')
    if which_bert=='bert':
        print('Using english based BERT model')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif which_bert=='ml-bert':
        print('Using multilingual based BERT model')
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    elif which_bert=='roberta':
        print('Using english based RoBERTa model')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        print('Using multilingual based RoBERTa model')
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
#     print('Gathering hyper-parameters')
#     dropout = float(input('Enter value of drop-out (0 to 1): '))
#     EPOCHS = int(input('Enter number of epochs to train the model (default use 20): '))
#     LR = float(input('Enter learning rate (default use 0.000001): '))
#     batch_size = int(input('Enter the size of the batch in training (if you are facing memory error, try to lower the batch size, default batch size 8): '))
    num_class=len(train_data['target'].unique())
    model = BertClassifier(dropout,which_bert,num_class)
#     pretrain_model = input('Do you want to load pre-trained network? (1-Yes, 2-No): ')
    if len(pretrain_model)>1:
#         which_path = input('Provide path of the pre-trained network: ')
        model.load_state_dict(torch.load(pretrain_model))
    
    train(model, train_data, valid_data, tokenizer, batch_size, LR, EPOCHS)
    print('Training has been completed successfully! Please check log.csv file for logs.')