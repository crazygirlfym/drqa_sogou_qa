#--*- coding:utf-8 -*--
import json
import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
def loadExpandDict(expand_file):
    expand = open(expand_file)
    answer_dict = {}

    for line in expand:
        line = line.strip()
        qid, answer, answer_expand = line.strip().split("\t")
        answer_expand = set(answer_expand.split("|"))
        answer_expand.add(answer)
        answer_dict[qid] = list(answer_expand)

    return answer_dict


def pointLabel(golden_answer, evidence_str, evidence_tokens, evidence_charstart):
    """
      :param golden_answer: List of answers, maybe the answer is not unique
      :param evidence_str:  evidence before  segmentation
      :param evidence_tokens:  evidence tokens after segmentation
      :param evidence_charstart: seg offset
      :return: starts [], ends[]
    """
    ## 如:  陶渊明   陶渊明 诗人
    ## 如： 双十协议  双十  协议
    ## 如：  桂花   咏桂花

    ## [['no_answer']]
    flag =False
    if len(golden_answer) == 1 and (golden_answer[0]) == 'no_answer':
        return [-1], [-1]
    else:
        starts = []
        ends = []
        # print ("test....")
        for answer in golden_answer:
            # print ("###########################3")
            # print answer
            start_indexs = findsubstr(evidence_str, answer)
            end_indexs = cal_endindex(answer, start_indexs)
            # print start_indexs
            # print end_indexs

            for idx, index_start in enumerate(start_indexs):

                index_end = int(end_indexs[idx])
                # print evidence_str[index_start:index_end+1]
                for i in range(len(evidence_tokens)):

                    if int(evidence_charstart[i]) <= index_start and index_start < len(evidence_tokens[i]) + int(evidence_charstart[i]):
                        if not i in starts:

                            starts.append(i)
                            flag=False
                        else:
                            flag=True

                    if index_end < int(evidence_charstart[i]) + len(evidence_tokens[i]) and int(evidence_charstart[i]) <= index_end:
                            if not flag:
                                ends.append(i)
        return starts, ends


def findsubstr(strs, target):
    """

    :param strs:  the strs to find str(target)
    :param target: the str to be found
    :return: the offset list
    """
    start = 0
    result = []
    while True:
        index = strs.find(target, start)
        if index == -1:
            break
        result.append(index)
        start = index + 1

    return result


def cal_endindex(target, starts):
    ends = []
    n = len(target)
    for start in starts:
        ends.append(start + n - 1)
    return ends


def qecomm_features(question, evidence_tokens):
    featurelist = [0 for _ in range(len(evidence_tokens))]
    for idx in range(len(evidence_tokens)):
        token = evidence_tokens[idx]

        if token in question:
            featurelist[idx] = 1

    return featurelist

def lowerList(list):
    result_list = []

    for item in list:
        result_list.append(item.lower())

    return result_list


def formatsougoudata(filepath, orgin_data, expand_file, is_train, output_path):
    answer_dict = loadExpandDict(expand_file)
    all_data_dict = {}
    f = open(filepath)
    result = open(output_path, 'w')
    for line in f:
        # print(line)
        line = line.strip()
        linedict = {}
        json_str = json.loads(line,encoding="utf-8")

        q_key = json_str['query_id']
        question_tokens = json_str['question_tokens']
        question_tokens = lowerList(question_tokens)

        question_poss = json_str['question_poss']
        question_ners = json_str['question_ners']
        question_starts = json_str['question_starts']
        question_ends = json_str['question_ends']
        query = json_str['question']
        evidences = json_str['evidences']

        linedict['q_key'] = q_key
        linedict['question_tokens'] = question_tokens
        linedict['question_pos'] = question_poss
        linedict['question_ners'] = question_ners


        if is_train:
            answer = json_str['answer']
            answers =answer_dict[q_key]
            answers = lowerList(answers)
            answers.insert(0,answer)
            # for ans in answers:
            #     print (ans)
            # print ("_____answers_________")

        evidencelist = []
        frecounter = {}
        # print(answer)
        for evidence in evidences:
            evidencedict = {}
            evidence_ekey = evidence['e_key']
            evidence_tokens = evidence['tokens']
            evidence_tokens = lowerList(evidence_tokens)


            evidence_poss = evidence['poss']
            evidence_ners = evidence['ners']
            evidence_starts = evidence['starts']
            evidence_ends = evidence['ends']
            evidencedict['e_key'] = evidence_ekey
            evidencedict['evidence_tokens'] = evidence_tokens
            evidencedict['evidence_pos'] = evidence_poss
            evidencedict['evidence_ners'] = evidence_ners

            passage_text = evidence['text']





            for i in range(len(evidence_tokens)):

                if not evidence_poss[i] == 'PU':

                    frecounter.setdefault((evidence_tokens[i]), 0)
                    frecounter[evidence_tokens[i]] = frecounter[evidence_tokens[i]] + 1

            if is_train:
                b = True
                for ans in answers:
                    if ans in passage_text:
                        b = False
                        break
                if b:
                    starts = [-1]
                    ends = [-1]

                else:
                    answers = list(answers)
                    starts, ends = pointLabel(answers, passage_text, evidence_tokens, evidence_starts)
                # print(starts)
                # print(ends)
                # for i in range(len(starts)):
                #     print (" ".join(evidence_tokens[starts[i]:ends[i]+1]))
                # print ("-----------")
                # print(evidence_tokens)
                evidencedict['answer_starts'] = starts
                # print(evidence_tokens[starts[0]])
                evidencedict['answer_ends'] = ends

            qefeature = qecomm_features(query, evidence_tokens)
            evidencedict['qecomm'] = qefeature
            evidencedict['qecomm'] = qefeature
            evidencelist.append(evidencedict)

        for evi_dict in evidencelist:
            evidence_tokens = evi_dict['evidence_tokens']
            evidence_pos = evi_dict['evidence_pos']
            fre_tokens = []
            for i in range(len(evidence_tokens)):
                token = evidence_tokens[i]
                pos = evidence_pos[i]
                if pos == 'PU':
                    fre_tokens.append(0)
                else:
                    fre_tokens.append(frecounter[token])
            evi_dict['fre_tokens'] = fre_tokens
        linedict['evidences'] = evidencelist

        outputline = json.dumps(linedict)
        all_data_dict[q_key] = outputline

    f2 = open(orgin_data)
    for line in f2:
        line = line.strip()
        json_str = json.loads(line)
        queryid = str((json_str['query_id']))

        result.write(all_data_dict[queryid]+"\n")
        # print(all_data_dict[queryid])


def formatsougoudata2(filepath, orgin_data, is_train, output_path):

    all_data_dict = {}
    f = open(filepath)
    result = open(output_path, 'w')
    for line in f:
        # print(line)
        line = line.strip()
        linedict = {}
        json_str = json.loads(line, encoding="utf-8")

        q_key = json_str['query_id']
        question_tokens = json_str['question_tokens']
        question_tokens = lowerList(question_tokens)

        question_poss = json_str['question_poss']
        question_ners = json_str['question_ners']
        question_starts = json_str['question_starts']
        question_ends = json_str['question_ends']
        query = json_str['question']
        evidences = json_str['evidences']

        linedict['q_key'] = q_key
        linedict['question_tokens'] = question_tokens
        linedict['question_pos'] = question_poss
        linedict['question_ners'] = question_ners


        if is_train:
            answer = json_str['answer']

        evidencelist = []
        frecounter = {}
        # print(answer)
        for evidence in evidences:
            evidencedict = {}
            evidence_ekey = evidence['e_key']
            evidence_tokens = evidence['tokens']
            evidence_tokens = lowerList(evidence_tokens)

            evidence_poss = evidence['poss']
            evidence_ners = evidence['ners']
            evidence_starts = evidence['starts']
            evidence_ends = evidence['ends']
            evidencedict['e_key'] = evidence_ekey
            evidencedict['evidence_tokens'] = evidence_tokens
            evidencedict['evidence_pos'] = evidence_poss
            evidencedict['evidence_ners'] = evidence_ners

            passage_text = evidence['text']
            # print(passage_text)
            for i in range(len(evidence_tokens)):

                if not evidence_poss[i] == 'PU':

                    frecounter.setdefault((evidence_tokens[i]), 0)
                    frecounter[evidence_tokens[i]] = frecounter[evidence_tokens[i]] + 1


            qefeature = qecomm_features(query, evidence_tokens)
            evidencedict['qecomm'] = qefeature
            evidencedict['qecomm'] = qefeature
            evidencelist.append(evidencedict)

        for evi_dict in evidencelist:
            evidence_tokens = evi_dict['evidence_tokens']
            evidence_pos = evi_dict['evidence_pos']
            fre_tokens = []
            for i in range(len(evidence_tokens)):
                token = evidence_tokens[i]
                pos = evidence_pos[i]
                if pos == 'PU':
                    fre_tokens.append(0)
                else:
                    fre_tokens.append(frecounter[token])
            evi_dict['fre_tokens'] = fre_tokens
        linedict['evidences'] = evidencelist

        outputline = json.dumps(linedict)
        all_data_dict[q_key] = outputline

    f2 = open(orgin_data)
    for line in f2:
        line = line.strip()
        json_str = json.loads(line)
        queryid = str((json_str['query_id']))

        result.write(all_data_dict[queryid]+"\n")
        # print(all_data_dict[queryid])






if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--type', dest='type_name', type=str, help='data type: train,validation,test')
    parser.add_argument('--input_file', dest='input_path', type=str, help='filepath for the input data')
    parser.add_argument('--output_file', dest='output_path', type=str, help='filepath for the output data')
    parser.add_argument('--origin_file', dest='origin_path', type=str, help='origin file by sougou')
    parser.add_argument('--expand_file', dest='expand_path', type=str, help='expand file by sougou')
    # fileinput = '/home/iscas/fym/codes/sougou_train.json'
    # output = './test.json'
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    type_name = args.type_name
    origin_path = args.origin_path
    expand_path = args.expand_path

    if type_name == 'train' or type_name == 'validation':
        formatsougoudata(input_path, origin_path,expand_path,True, output_path)
    elif type_name =="test":
        formatsougoudata2(input_path, origin_path, False, output_path)
    else:
        print("error!!!")


    # python json_read.py --input_file=/media/iscas/linux/fym/data/sogou_java_valid --output_file=/media/iscas/linux/fym/data/sogou_valid_final --origin_file=/media/iscas/linux/fym/data/sogou_valid.json --expand_file=/media/iscas/linux/fym/data/qid_answer_expand/qid_answer_expand.valid --type=train

#python json_read.py --input_file=/media/iscas/linux/fym/data/java_pre_data/train_factoid_2_java.json --output_file=/media/iscas/linux/fym/data/sogou_train_factoid_2 --origin_file=/media/iscas/linux/fym/data/raw_data/train_factoid_2.json --expand_file=/media/iscas/linux/fym/data/qid_answer_expand/qid_answer_expand.train.second --type=train


