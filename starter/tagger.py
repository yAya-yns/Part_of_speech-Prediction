# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)

'''
Planning:

Step 0. Read files

Step 1. construct a data class
    - segment sentence based on punctuation
    - for train data:
        - segment word & tag into list of words
    for test data:
        - segment word into list of words

Step 2. Contruct model class
    - train model
        - i.e. create 3 tables
            a. initial probability table (unigram)
            b. transition probability table (biagram) -> extended to triagram for improved version
            c. emission probability table (unigram, a frequentist style inferencing)
                - including "unseen" word as an option
    - inferencing:
        - first label all special word with tag
            - ex. punctuation
            * we can use these special words with high certainty as a landmark 
                - similar idea as Kalman Filter
        - use 3 tables above and viterbi algorithm to inference the rest

Step 3. Combine inferencing result and output as a file
'''

import os
import sys
import random
import numpy as np
# from tqdm import tqdm  # to be deleted! 

def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")
    #
    # YOUR IMPLEMENTATION GOES HERE
    #
    data = dataset(training_list, test_file)
    data.preprocessing(shuffle=False)
    # data.train_sentences will be the training dataset: a list of sentence, each sentence is a list of word + tag in term
    # data.train_sentences will be the testing dataset
    model = HMM()
    model.dataloader(data)
    print("Training")
    model.train()
    print("Inferencing the Test File")
    model.inference(optimize=True)

    return True


class dataset:
    def __init__(self, training_list, test_file):
        self.training_file_name_list = training_list
        self.test_file_name = test_file
        self.train_sentences_n_tags = None  # will be added by self.sentence_segmentation()
        self.test_sentences = None  # will be added by self.sentence_segmentation()
        self.train_sentences = None
        self.train_tags = None


    def preprocessing(self, shuffle=False):
        self.read_files()
        self.sentence_segmentation(shuffle=shuffle)
        self.split_labels()
        return True


    def read_files(self):
        self.train_files_list = []

        for f in self.training_file_name_list:
            self.train_files_list.append(open(f, "r").readlines())
        
        self.test_file = open(self.test_file_name, "r").readlines()
        return True


    def sentence_segmentation(self, shuffle):
        # input: self.train_files_list, self.test_file
        # output: self.train_sentence, self.test_sentence
        # self.train_sentences_n_tags will be the training dataset: a list of sentence, each sentence is a list of "token : tag" in term
        # self.test_sentences will be the testing dataset: a list of sentence, each sentence is a list of "token"
        # segments by punctuation
        end_of_sentence = ".?!"

        self.train_sentences_n_tags = []
        for training_file in self.train_files_list:
            sentence = []
            for word in training_file:  # merging all training lines into one list
                if len(word) == 0:
                    continue
                sentence.append(word)
                if word[0] in end_of_sentence:
                    self.train_sentences_n_tags.append(sentence)  # sentence complete
                    sentence = []  # clear sentence
            if len(sentence) != 0:  # if file finished without a punctuation break, still add to the end of sentence.
                self.train_sentences_n_tags.append(sentence)
        if shuffle == True:
            random.shuffle(self.train_sentences_n_tags)

        self.test_sentences = []
        sentence = []
        for word in self.test_file:  # merging all training lines into one list
            if len(word) == 0:
                    continue
            sentence.append(word.rstrip("\n"))
            if word[0] in end_of_sentence:
                self.test_sentences.append(sentence)  # sentence complete
                sentence = []  # clear sentence
        if len(sentence) != 0:  # if file finished without a punctuation break, still add to the end of sentence.
            self.test_sentences.append(sentence)
        
        return True


    def split_labels(self):
        # input: self.train_sentences_n_tags
        # output: self.train_sentences and self.train_tags
        self.train_sentences = []
        self.train_tags = []

        for sentence in self.train_sentences_n_tags:
            temp_sentence = []
            temp_tags = []
            for word in sentence:
                word = word.split(" : ")
                if len(word) == 2:  # more than just " : "
                    word[1] = word[1].rstrip("\n")
                    temp_sentence.append(word[0])
                    temp_tags.append(word[1])
            if len(temp_sentence) > 1 and len(temp_tags) > 1:
                self.train_sentences.append(temp_sentence)
                temp_sentence = []
                self.train_tags.append(temp_tags)
                temp_tags = []

        return True


class HMM:
    def __init__(self):
        return None

    def dataloader(self, data):
        # input, data class
        self.data = data
        self.num_of_tag_type = 0
        self.num_of_word_type = 0
        self.possible_tags = []
        

    def train(self):
        count = {}
        emission = {}
        transition = {}

        train_sentences = self.data.train_sentences
        train_tags = self.data.train_tags

        # count 
        for i in range(len(train_sentences)):
            if "<SOS>" in count.keys() and "<EOS>" in count.keys():  # for each sentence, count start of sentence, and end of sentence
                count["<SOS>"] += 1
                count["<EOS>"] += 1
            else:
                count["<SOS>"] = 1
                count["<EOS>"] = 1

            for j in range(len(train_sentences[i])):
                if train_sentences[i][j] in count.keys():  # i_th sentence, j_th WORD
                    count[train_sentences[i][j]] += 1
                else:
                    count[train_sentences[i][j]] = 1
                    self.num_of_word_type += 1
                
                if train_tags[i][j] in count.keys():  # i_th sentence, j_th TAG
                    count[train_tags[i][j]] += 1
                else:
                    count[train_tags[i][j]] = 1
                    self.num_of_tag_type += 1
                    self.possible_tags.append(train_tags[i][j])
        if "<SOS>" in self.possible_tags:
            self.possible_tags.pop("<SOS>")
        if "<EOS>" in self.possible_tags:
            self.possible_tags.pop("<EOS>")

        # transition & Emission
        for i in range(len(train_sentences)):
            last_tag = "<SOS>"  # start of sentence
            for j in range(len(train_sentences[i])):
                cur_word = train_sentences[i][j]
                cur_tag = train_tags[i][j]
                # transition
                if (last_tag, cur_tag) in transition.keys():
                    transition[(last_tag, cur_tag)] += 1
                else:
                    transition[(last_tag, cur_tag)] = 1
                # emission
                if (cur_tag, cur_word) in emission.keys():
                    emission[(cur_tag, cur_word)] += 1
                else:
                    emission[(cur_tag, cur_word)] = 1
                last_tag = cur_tag
            # finishing up a sentence
            if (last_tag, "<EOS>") in transition.keys():  # end of sentence
                    transition[(last_tag, "<EOS>")] += 1
            else:
                transition[(last_tag, "<EOS>")] = 1

       
        emission = self.normalization(count, emission)  # convert from tally to probability
        transition = self.normalization(count, transition)  # convert from tally to probability
        self.emission = emission  # store as model parameter
        self.transition = transition  # store as model parameter
        return True

    
    def normalization(self, count, input_dict):
        output_dict = {}
        for term1, term2 in input_dict.keys():
            output_dict[(term1, term2)] = input_dict[(term1, term2)] / count[term1]
        return output_dict


    def inference(self, optimize=False):
        # optimize = True
        with open(output_file, 'w') as f:
            for sentence in (self.data.test_sentences):
                # print(sentence)
                tags_list = self.viterbi(sentence)
                # print(tags_list)
                
                # a = input()
                # tags_list = ["abc"] * len(sentence)  # for testing
                if optimize == True:
                    tags_list = self.optimization(sentence, tags_list)
                for i in range(len(sentence)):
                    f.write((sentence[i] + " : " + tags_list[i] + "\n"))

        return True


    def viterbi(self, sentence):
        # opt_prob = {(0, "<SOS>"): 0}
        # opt_path = {(0, "<SOS>"): None}

        # last = "<SOS>"
        # for cur in self.possible_tags:
        #     if (0, "<SOS>") in opt_prob and (last, cur) in transition.keys():
        #         prob = opt_prob[last]
        
        # creating matrix 

        prob_trellis = []
        for i in range(self.num_of_tag_type):
            prob_trellis.append([None for i in range(len(sentence))])

        path_trellis = []
        for i in range(self.num_of_tag_type):
            path_trellis.append([None for i in range(len(sentence))])

        for s in range(self.num_of_tag_type):
            # print(self.transition_filter("<SOS>", self.possible_tags[s]) * self.emission_filter(self.possible_tags[s], sentence[0]))

            prob_trellis[s][0] = self.transition_filter("<SOS>", self.possible_tags[s]) * self.emission_filter(self.possible_tags[s], sentence[0])
            path_trellis[s][0] = [s]
       
        prob_trellis = self.normalize_prob_trellis(prob_trellis, 0)

        for o in range(1, len(sentence)):
            for s in range(self.num_of_tag_type):

                x_list = [prob_trellis[x][o-1] * self.transition_filter(self.possible_tags[x], self.possible_tags[s]) * self.emission_filter(self.possible_tags[s], sentence[o]) for x in range(self.num_of_tag_type)]
                np_x_list = np.array(x_list)
                x = np.argmax(np_x_list)

                # np_prob_trellis = np.array(prob_trellis)
                # x = np.argmax(np_prob_trellis[:, o-1])
                

                # Max = prob_trellis[0][o-1]  # initialization to find argmax 
                # argMax = 0  # initialization to find argmax 
                # for x in range(1, self.num_of_tag_type):
                #     prob = prob_trellis[x][o-1] * self.transition_filter(self.possible_tags[x], self.possible_tags[s]) * self.emission_filter(self.possible_tags[s], sentence[o])
                #     if prob > Max:
                #         Max = prob
                #         argMax = x
                # x = argMax
                # # print(prob_trellis[:][o-1])
                # print(x,y)
                # xxxx

                # prob_trellis[s][o] = prob_trellis[x][o-1] * self.transition_filter(self.possible_tags[x], self.possible_tags[s]) * self.emission_filter(self.possible_tags[s], sentence[o])
                prob_trellis[s][o] = max(x_list)
                
                
                # prob_trellis[s][o] = max(prob_trellis[s][o], 1e-10)

                # print(type(path_trellis[x][o-1]))
                # if type(path_trellis[x][o-1]) != list:
                #     print(s, x, o-1)
                #     print(path_trellis[x])
                #     xxx
                #  some of them is not list type.  but rather none type.  not the firs one
                updated_path = list(path_trellis[x][o-1])
                updated_path.append(s)
                path_trellis[s][o] = updated_path

            
            
            prob_trellis = self.normalize_prob_trellis(prob_trellis, o)
            # if o == 5:
            #     print(prob_trellis[:5])

        # print("-----------")
        # prob_trellis = self.normalize_prob_trellis_print(prob_trellis, -1)
        
        # argMax = 0
        # Max = 0
        # for i in range(len(prob_trellis)):
        #     if prob_trellis[i][-1] > Max:
        #         argMax = i
        np_prob_trellis = np.array(prob_trellis)
        argMax = np.argmax(np_prob_trellis[:, -1])
        best_path = path_trellis[argMax][-1]

        tags_list = []
        for tag_idx in best_path:
            tags_list.append(self.possible_tags[tag_idx])
        return tags_list


    def emission_filter(self, tag, word):
        # to avoid 0 prob, we arbitrarily add value
        if (tag, word) in self.emission:
            return self.emission[(tag, word)]
        else: # unseen emission
            # return 1/self.num_of_word_type
            return 1e-10

    def transition_filter(self, last_tag, cur_tag):
        # to avoid 0 prob, we arbitrarily add value
        if (last_tag, cur_tag) in self.transition:
            return self.transition[(last_tag, cur_tag)]
        else: # unseen emission
            # return 1/self.num_of_tag_type
            return 1e-10

    def normalize_prob_trellis(self, prob_trellis, col):
        accumulator = 0
        for row in prob_trellis:
            accumulator += row[col]
        for row in prob_trellis:
            row[col] /= accumulator
        return prob_trellis

    def normalize_prob_trellis_print(self, prob_trellis, col):
        accumulator = 0
        print(prob_trellis[:5])
        for row in prob_trellis:
            accumulator += row[col]
        for row in prob_trellis:
            row[col] /= accumulator

        print(prob_trellis[:5])
        return prob_trellis

    def optimization(self, sentence, tags_list):
        for i in range(len(sentence)):
            if sentence[i] in opt.keys():
                tags_list[i] = opt[sentence[i]]
        return tags_list

opt = {
        ".": "PUN",
        "?": "PUN",
        "!": "PUN",
        ",": "PUN",
        ";": "PUN",
        '"': "PUQ",
        "(": "PUL",
        ")": "PUR",
        "\'": "POS",
        "of": "PRF",
        "the": "AT0",
        "The": "ATO",
        'a' : 'AT0',
        "He": "PNP",
        "he": "PNP",
        "She": "PNP",
        "she": "PNP",
        "I" : "PNP",
        "and": "CJC",
        "but": "CJC"
        }   


if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    # print("Training files: " + str(training_list))
    # print("Test file: " + test_file)
    # print("Ouptut file: " + output_file)

    # Start the training and tagging operation.
    
    result = tag (training_list, test_file, output_file)
    