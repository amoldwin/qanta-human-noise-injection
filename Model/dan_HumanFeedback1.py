import argparse
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import json
import time
import nltk

import pickle

kUNK = '<unk>'
kPAD = '<pad>'
MODEL_PATH = 'cucumber_dan.pt'
PARAM_PATH = 'cucumber_params.pickle'

# TODO
def class_labels(data):
    class_to_i = {}
    i_to_class = {}
    i = 0
    for item in data:
        ans = item[1]
        if ans not in class_to_i.keys():
            class_to_i[ans] = i
            i_to_class[i] = ans
            i+=1
    return class_to_i, i_to_class

# TODO
def load_data(filename, lim):
    """
    load the json file into data list
    """

    data = list()
    with open(filename) as json_data:
        if lim>0:
            questions = json.load(json_data)["questions"][:lim]
        else:
            questions = json.load(json_data)["questions"]
        for q in questions:
            if filename == "../Preprocessing/trainingData.json":
                q = questions[q]
                q_text = nltk.word_tokenize(q['text'])
                label = q['answer']
            else:
                q_text = nltk.word_tokenize(q['text'])
                label = q['page']
            if "guesses" in q:
                guess = q["guesses"]
                if label:
                    data.append((q_text, label, guess))
            elif label:
                data.append((q_text, label))
    return data

# TODO
def load_words(exs):
    """
    vocabuary building
    Keyword arguments:
    exs: list of input questions-type pairs
    """

    words = set()
    word2ind = {kPAD: 0, kUNK: 1}
    ind2word = {0: kPAD, 1: kUNK}
    for q_text, _, _ in exs:
        for w in q_text:
            words.add(w)
    words = sorted(words)
    for w in words:
        idx = len(word2ind)
        word2ind[w] = idx
        ind2word[idx] = w
    words = [kPAD, kUNK] + words
    return words, word2ind, ind2word


class QuestionDataset(Dataset):
    """
    Pytorch data class for questions
    """

    ###TODO
    def __init__(self, examples, word2ind, num_classes, class2ind=None, three=False):
        self.questions = []
        self.labels = []
        self.guesses = []
        self.three = three
        if three:
            # print("THREE")
            for qq, ll, gg in examples:
                # print(qq)
                # print(ll)
                # print(gg)
                self.questions.append(qq)
                self.labels.append(ll)
                self.guesses.append(gg)
            if type(self.labels[0])==str:
                for i in range(len(self.labels)):
                    try:
                        self.labels[i] = class2ind[self.labels[i]]
                    except:
                        self.labels[i] = num_classes

                    new_guesses = {}
                    for guess in self.guesses[i]:
                        try:
                            class_num = class2ind[guess]
                        except:
                            print("UNK CLASS _____________________")
                            class_num = num_classes
                        new_guesses[class_num] = self.guesses[i][guess]
                    self.guesses[i] = new_guesses
        else:
            for qq, ll in examples:
                self.questions.append(qq)
                self.labels.append(ll)
        
            if type(self.labels[0])==str:
                for i in range(len(self.labels)):
                    try:
                        self.labels[i] = class2ind[self.labels[i]]
                    except:
                        self.labels[i] = num_classes

        self.word2ind = word2ind
    
    ###TODO
    def __getitem__(self, index):
        if self.three:
            return self.vectorize(self.questions[index], self.word2ind), \
              self.labels[index], self.guesses[index]
        else:
             return self.vectorize(self.questions[index], self.word2ind), \
              self.labels[index]
    
    ###TODO
    def __len__(self):
        return len(self.questions)

    @staticmethod
    def vectorize(ex, word2ind):
        """
        vectorize a single example based on the word2ind dict. 
        Keyword arguments:
        exs: list of input questions-type pairs
        ex: tokenized question sentence (list)
        label: type of question sentence
        Output:  vectorized sentence(python list) and label(int)
        e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
        """
        vec_text = [0] * len(ex)

        #### modify the code to vectorize the question text
        #### You should consider the out of vocab(OOV) cases
        #### question_text is already tokenized    
        ####Your code here
        for i, word in enumerate(ex):
            if word not in word2ind:
                vec_text[i] = word2ind[kUNK]
            else:
                vec_text[i] = word2ind[word]

        return vec_text

    
###TODO

def batchify(batch):
    """
    Gather a batch of individual examples into one batch, 
    which includes the question text, question length and labels 
    Keyword arguments:
    batch: list of outputs from vectorize function
    """

    question_len = list()
    label_list = list()
    guess_list = list()
    three = False
    for ex in batch:
        if three or len(ex) == 3:
            three = True
        # print("BBBBBB")
        # print(ex)
        question_len.append(len(ex[0]))
        label_list.append(ex[1])
        if three:
            guess_list.append(ex[2])

    target_labels = torch.cuda.LongTensor(label_list)
    x1 = torch.cuda.LongTensor(len(question_len), max(question_len)).zero_()
    for i in range(len(question_len)):
        question_text = batch[i][0]
        vec = torch.cuda.LongTensor(question_text)
        x1[i, :len(question_text)].copy_(vec)

    if three:
        q_batch = {'text': x1, 'len': torch.cuda.FloatTensor(question_len), 'labels': target_labels, 'guesses': guess_list}
    else:
        q_batch = {'text': x1, 'len': torch.cuda.FloatTensor(question_len), 'labels': target_labels}
    return q_batch


def evaluate(data_loader, model, device):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """

    model.eval()
    num_examples = 0
    error = 0
    for idx, batch in enumerate(data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        ####Your code here
        logits = model(question_text, question_len)
        ### END of my code

        top_n, top_i = logits.topk(1)
        num_examples += question_text.size(0)
        error += torch.nonzero(top_i.squeeze() - torch.cuda.LongTensor(labels)).size(0)
    accuracy = 1 - error / num_examples
    print('accuracy', accuracy)
    return accuracy


def train(args, model, train_data_loader, dev_data_loader, accuracy, device):
    """
    Train the current model
    Keyword arguments:
    args: arguments 
    model: model to be trained
    train_data_loader: pytorch build-in data loader output for training examples
    dev_data_loader: pytorch build-in data loader output for dev examples
    accuracy: previous best accuracy
    device: cpu of gpu
    """

    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()

    #### TODO

    for idx, batch in enumerate(train_data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']
        guesses =  batch['guesses']

        #### TODO
        model.zero_grad()
        out = model(question_text, question_len) #in video, it's model(batches), where "batches" was not specified in the code
        loss = criterion(out, labels)
        multiplier = 0
        out_list = out.tolist()
        for i in range(len(labels)):
            # max_guess = (max(out_list[i]))
            # index = out_list[i].index(max_guess)
            max_guess, index = out[i].topk(1)
            if index in guesses[i]:
                multiplier += max_guess

        multiplier /= len(labels)
        loss.data = loss.data * (1-multiplier) 
        #Variation - smoothing by adding 5 and dividing by 6 (1-mult+5)/6 or something of that type (+x, /(x+1))
        loss.backward()
        optimizer.step()
        ### TODO

        clip_grad_norm_(model.parameters(), args.grad_clipping) 
        print_loss_total += loss.data.cpu().numpy()
        epoch_loss_total += loss.data.cpu().numpy()

        if idx % args.checkpoint == 0 and idx > 0:
            print_loss_avg = print_loss_total / args.checkpoint

            print('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
            print_loss_total = 0
            curr_accuracy = evaluate(dev_data_loader, model, device)
            if accuracy < curr_accuracy:
                torch.save(model, args.save_model)
                accuracy = curr_accuracy
    return accuracy




class DanModel(nn.Module):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    #### You don't need to change the parameters for the model for passing tests, might need to tinker to improve performance/handle
    #### pretrained word embeddings/for your project code.


    def __init__(self, n_classes, vocab_size, emb_dim=50,
                 n_hidden_units=50, nn_dropout=.5):
        super(DanModel, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.linear1 = nn.Linear(emb_dim, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_classes)

        # Create the actual prediction framework for the DAN classifier.

        # You'll need combine the two linear layers together, probably
        # with the Sequential function.  The first linear layer takes
        # word embeddings into the representation space, and the
        # second linear layer makes the final prediction.  Other
        # layers / functions to consider are Dropout, ReLU. 
        # For test cases, the network we consider is - linear1 -> ReLU() -> Dropout(0.5) -> linear2

        #### TODO
        self.classifier = nn.Sequential(self.linear1, nn.ReLU(), nn.Dropout(nn_dropout), self.linear2)
        self._softmax = nn.Softmax()
        ### TODO
        
       
    def forward(self, input_text, text_len, is_prob=False):
        """
        Model forward pass, returns the logits of the predictions.
        
        Keyword arguments:
        input_text : vectorized question text 
        text_len : batch * 1, text length for each question
        is_prob: if True, output the softmax of last layer
        """

        logits = torch.cuda.LongTensor([0.0] * self.n_classes)

        # Complete the forward funtion.  First look up the word embeddings.
        text_embed = self.embeddings(input_text)
        # Then average them 
        encoded = text_embed.sum(1)
        encoded /= text_len.view(text_embed.size(0), -1)
        # Before feeding them through the network
        
        logits = self.classifier(encoded)

        if is_prob:
            logits = self._softmax(logits)

        return logits



# You basically do not need to modify the below code 
# But you may need to add funtions to support error analysis 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question Type')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--train-file', type=str, default='../Preprocessing/trainingData.json')
    parser.add_argument('--dev-file', type=str, default='../Preprocessing/qanta.dev.json')
    parser.add_argument('--test-file', type=str, default='../Preprocessing/qanta.test.json')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--grad-clipping', type=int, default=5)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--save-model', type=str, default=MODEL_PATH)
    parser.add_argument('--load-model', type=str, default=MODEL_PATH)
    parser.add_argument("--limit", help="Number of training documents", type=int, default=-1, required=False)
    parser.add_argument('--checkpoint', type=int, default=50)

    args = parser.parse_args()
    #### check if using gpu is available
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    ### Load data
    train_exs = load_data(args.train_file, args.limit)
    dev_exs = load_data(args.dev_file, -1)
    test_exs = load_data(args.test_file, -1)

    ### Create vocab
    voc, word2ind, ind2word = load_words(train_exs)

    #get num_classes from training + dev examples - this can then also be used as int value for those test class labels not seen in training+dev.
    num_classes = len(list(set([ex[1] for ex in train_exs+dev_exs])))

    print(num_classes)

    #get class to int mapping
    class2ind, ind2class = class_labels(train_exs + dev_exs)  

    #SAVE THINGS
    print("SAVING PARAMETERS")
    with open(PARAM_PATH, 'wb') as f:
        pickle.dump({
            'ind2class': ind2class,
            'word2ind': word2ind,
        }, f)

    if args.test:
        model = torch.load(args.load_model)
        #### Load batchifed dataset
        test_dataset = QuestionDataset(test_exs, word2ind, num_classes, class2ind, False)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                               sampler=test_sampler, num_workers=0,
                                               collate_fn=batchify)
        evaluate(test_loader, model, device)
    else:
        if args.resume:
            model = torch.load(args.load_model)
        else:
            model = DanModel(num_classes, len(voc))
            model.to(device)
        print(model)
        #### Load batchifed dataset
        train_dataset = QuestionDataset(train_exs, word2ind, num_classes, class2ind, True)
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        dev_dataset = QuestionDataset(dev_exs, word2ind, num_classes, class2ind, False)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                               sampler=dev_sampler, num_workers=0,
                                               collate_fn=batchify)
        accuracy = 0
        for epoch in range(args.num_epochs):
            print('start epoch %d' % epoch)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler, num_workers=0,
                                               collate_fn=batchify)
            accuracy = train(args, model, train_loader, dev_loader, accuracy, device)
        print('start testing:\n')

        test_dataset = QuestionDataset(test_exs, word2ind, num_classes, class2ind)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                               sampler=test_sampler, num_workers=0,
                                               collate_fn=batchify)
        evaluate(test_loader, model, device)
