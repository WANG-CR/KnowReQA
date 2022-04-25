import torch
import random
import copy
import re
import numpy as np


class ReqaDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 args, 
                 question_insts, 
                 answer_insts):
        self.args = args
        self.question_insts = question_insts
        self.answer_insts = answer_insts

    def __len__(self):
        return len(self.question_insts)

    def __getitem__(self, idx):
        return self.question_insts[idx], self.answer_insts[idx], self.args


class NliDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 args, 
                 anchor_insts, 
                 pos_insts,
                 neg_insts=None):
        self.args = args
        self.anchor_insts = anchor_insts
        self.pos_insts = pos_insts
        self.neg_insts = neg_insts

    def __len__(self):
        return len(self.anchor_insts)

    def __getitem__(self, idx):
        if self.neg_insts is not None:
            return self.anchor_insts[idx], self.pos_insts[idx], self.neg_insts[idx], self.args
        return self.anchor_insts[idx], self.pos_insts[idx], self.args


class SentDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 args, 
                 sentence_insts):
        self.args = args
        self.sentence_insts = sentence_insts

    def __len__(self):
        return len(self.sentence_insts)

    def __getitem__(self, idx):
        return self.sentence_insts[idx], self.args


def reqa_fn(data):
    args = data[-1][-1]
    question_insts, answer_insts, _ = list(zip(*data))
    question_insts = padding_fn(question_insts, args.tokenizer, args.max_question_len)
    answer_insts = padding_fn(answer_insts, args.tokenizer, args.max_answer_len)

    return (*question_insts, *answer_insts)

def reqa_luke_fn(data):
    question_tuples, answer_tuples, _ = list(zip(*data))
    q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask = seperating_luke_fn(question_tuples)
    a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask = seperating_luke_fn(answer_tuples)
    return q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask, a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask
    
def train_fn_reqa():
    pass

def eval_fn_reqa():
    pass


def nli_fn(data):
    args = data[-1][-1]
    data = list(zip(*data))
    if len(data) == 3:
        anchor_insts, pos_insts, _ = data
        anchor_insts = padding_fn(anchor_insts, args.tokenizer, args.max_anchor_len)
        pos_insts = padding_fn(pos_insts, args.tokenizer, args.max_pos_len)
        return (*anchor_insts, *pos_insts)
    elif len(data) == 4:
        anchor_insts, pos_insts, neg_insts, _ = data
        anchor_insts = padding_fn(anchor_insts, args.tokenizer, args.max_anchor_len)
        pos_insts = padding_fn(pos_insts, args.tokenizer, args.max_pos_len)
        neg_insts = padding_fn(neg_insts, args.tokenizer, args.max_pos_len)
        return (*anchor_insts, *pos_insts, *neg_insts)


def sent_fn(data):
    args = data[-1][-1]
    sentence_insts, _ = list(zip(*data))
    sentence_insts = padding_fn(sentence_insts, args.tokenizer)
    return sentence_insts

def sent_luke_fn(data):
    sentence_tuples, _ = list(zip(*data))
    input_ids, entity_ids, entity_position_ids, attention_mask, entity_attention_mask = seperating_luke_fn(sentence_tuples)
    return input_ids, entity_ids, entity_position_ids, attention_mask, entity_attention_mask

def padding_fn(insts, tokenizer, max_len=-1):
    PAD = tokenizer.convert_tokens_to_ids('[PAD]')
    CLS = tokenizer.convert_tokens_to_ids('[CLS]')
    SEP = tokenizer.convert_tokens_to_ids('[SEP]')
    if not isinstance(insts, list):
        insts = list(insts)
    if max_len != -1:
        for i, inst in enumerate(insts):
            insts[i] = [CLS] + inst[:max_len-2] + [SEP]
    else:
        for i, inst in enumerate(insts):
            insts[i] = [CLS] + inst[:510] + [SEP]
            if max_len < len(insts[i]):
                max_len = len(insts[i])
        if max_len > 512:
            max_len = 512

    batch_seq = np.array([inst + [PAD] * (max_len - len(inst)) for inst in insts], dtype=np.int)
    batch_mask = np.array([[1] * len(inst) + [PAD] * (max_len-len(inst)) for inst in insts], dtype=np.int)
    batch_seq = torch.LongTensor(batch_seq)
    batch_mask = torch.LongTensor(batch_mask)

    return batch_seq, batch_mask

def seperating_luke_fn(tuples):
    input_ids = []
    entity_ids = []
    entity_position_ids = []
    attention_mask = []
    entity_attention_mask = []
    if not isinstance(tuples, list):
        tuples = list(tuples)
    for i, tuple in enumerate(tuples):
        input_ids.append(tuple["input_ids"])
        entity_ids.append(tuple["entity_ids"])
        entity_position_ids.append(tuple["entity_position_ids"])
        attention_mask.append(tuple["attention_mask"])
        entity_attention_mask.append(tuple["entity_attention_mask"])

    input_ids = torch.LongTensor(input_ids)
    entity_ids = torch.LongTensor(entity_ids)
    entity_position_ids = torch.LongTensor(entity_position_ids)
    attention_mask = torch.LongTensor(attention_mask)
    entity_attention_mask = torch.LongTensor(entity_attention_mask)
    return input_ids, entity_ids, entity_position_ids, attention_mask, entity_attention_mask



def do_sentence_breaks(uni_text):
    """Uses regexp substitution rules to insert newlines as sentence breaks.

    Args:
    uni_text: A (multi-sentence) passage of text, in Unicode.

    Returns:
    A Unicode string with internal newlines representing the inferred sentence
    breaks.
    """

    # The main split, looks for sequence of:
    #   - sentence-ending punctuation: [.?!]
    #   - optional quotes, parens, spaces: [)'" \u201D]*
    #   - whitespace: \s
    #   - optional whitespace: \s*
    #   - optional opening quotes, bracket, paren: [['"(\u201C]?
    #   - upper case letter or digit
    txt = re.sub(r'''([.?!][)'" %s]*)\s(\s*[['"(%s]?[A-Z0-9])''' % ('\u201D', '\u201C'),
               r'\1\n\2',
               uni_text)

    # Wiki-specific split, for sentence-final editorial scraps (which can stack):
    #  - ".[citation needed]", ".[note 1] ", ".[c] ", ".[n 8] "
    txt = re.sub(r'''([.?!]['"]?)((\[[a-zA-Z0-9 ?]+\])+)\s(\s*['"(]?[A-Z0-9])''',
               r'\1\2\n\4', txt)

    # Wiki-specific split, for ellipses in multi-sentence quotes:
    # "need such things [...] But"
    txt = re.sub(r'(\[\.\.\.\]\s*)\s(\[?[A-Z])', r'\1\n\2', txt)

    # Rejoin for:
    #   - social, military, religious, and professional titles
    #   - common literary abbreviations
    #   - month name abbreviations
    #   - geographical abbreviations
    #
    txt = re.sub(r'\b(Mrs?|Ms|Dr|Prof|Fr|Rev|Msgr|Sta?)\.\n', r'\1. ', txt)
    txt = re.sub(r'\b(Lt|Gen|Col|Maj|Adm|Capt|Sgt|Rep|Gov|Sen|Pres)\.\n',
               r'\1. ',
               txt)
    txt = re.sub(r'\b(e\.g|i\.?e|vs?|pp?|cf|a\.k\.a|approx|app|es[pt]|tr)\.\n',
               r'\1. ',
               txt)
    txt = re.sub(r'\b(Jan|Aug|Oct|Nov|Dec)\.\n', r'\1. ', txt)
    txt = re.sub(r'\b(Mt|Ft)\.\n', r'\1. ', txt)
    txt = re.sub(r'\b([ap]\.m)\.\n(Eastern|EST)\b', r'\1. \2', txt)

    # Rejoin for personal names with 3,2, or 1 initials preceding the last name.
    txt = re.sub(r'\b([A-Z]\.)[ \n]([A-Z]\.)[ \n]([A-Z]\.)[ \n]("?[A-Z][a-z])',
               r'\1 \2 \3 \4',
               txt)
    txt = re.sub(r'\b([A-Z]\.)[ \n]([A-Z]\.)[ \n]("?[A-Z][a-z])',
               r'\1 \2 \3',
               txt)
    txt = re.sub(r'\b([A-Z]\.[A-Z]\.)\n("?[A-Z][a-z])', r'\1 \2', txt)
    txt = re.sub(r'\b([A-Z]\.)\n("?[A-Z][a-z])', r'\1 \2', txt)

    # Resplit for common sentence starts:
    #   - The, This, That, ...
    #   - Meanwhile, However,
    #   - In, On, By, During, After, ...
    txt = re.sub(r'([.!?][\'")]*) (The|This|That|These|It) ', r'\1\n\2 ', txt)
    txt = re.sub(r'(\.) (Meanwhile|However)', r'\1\n\2', txt)
    txt = re.sub(r'(\.) (In|On|By|During|After|Under|Although|Yet|As |Several'
               r'|According to) ',
               r'\1\n\2 ',
               txt)

    # Rejoin for:
    #   - numbered parts of documents.
    #   - born, died, ruled, circa, flourished ...
    #   - et al (2005), ...
    #   - H.R. 2000
    txt = re.sub(r'\b([Aa]rt|[Nn]o|Opp?|ch|Sec|cl|Rec|Ecl|Cor|Lk|Jn|Vol)\.\n'
               r'([0-9IVX]+)\b',
               r'\1. \2',
               txt)
    txt = re.sub(r'\b([bdrc]|ca|fl)\.\n([A-Z0-9])', r'\1. \2', txt)
    txt = re.sub(r'\b(et al)\.\n(\(?[0-9]{4}\b)', r'\1. \2', txt)
    txt = re.sub(r'\b(H\.R\.)\n([0-9])', r'\1 \2', txt)

    # SQuAD-specific joins.
    txt = re.sub(r'(I Am\.\.\.)\n(Sasha Fierce|World Tour)', r'\1 \2', txt)
    txt = re.sub(r'(Warner Bros\.)\n(Records|Entertainment)', r'\1 \2', txt)
    txt = re.sub(r'(U\.S\.)\n(\(?\d\d+)', r'\1 \2', txt)
    txt = re.sub(r'\b(Rs\.)\n(\d)', r'\1 \2', txt)

    # SQuAD-specific splits.
    txt = re.sub(r'\b(Jay Z\.) ([A-Z])', r'\1\n\2', txt)
    txt = re.sub(r'\b(Washington, D\.C\.) ([A-Z])', r'\1\n\2', txt)
    txt = re.sub(r'\b(for 4\.\)) ([A-Z])', r'\1\n\2', txt)
    txt = re.sub(r'\b(Wii U\.) ([A-Z])', r'\1\n\2', txt)
    txt = re.sub(r'\. (iPod|iTunes)', r'.\n\1', txt)
    txt = re.sub(r' (\[\.\.\.\]\n)', r'\n\1', txt)
    txt = re.sub(r'(\.Sc\.)\n', r'\1 ', txt)
    txt = re.sub(r' (%s [A-Z])' % '\u2022', r'\n\1', txt)
    return txt

def infer_sentence_breaks(uni_text):
    """
    Generates (start, end) pairs demarking sentences in the text.
    """
    uni_text = re.sub(r'\n', r' ', uni_text)  # Remove pre-existing newlines.
    text_with_breaks = do_sentence_breaks(uni_text)
    starts = [m.end() for m in re.finditer(r'^\s*', text_with_breaks, re.M)]
    sentences = [s.strip() for s in text_with_breaks.split('\n')]
    assert len(starts) == len(sentences)
    for i in range(len(sentences)):
        start = starts[i]
        end = start + len(sentences[i])
        yield start, end