import json
import logging
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from transformers import BertModel, LukeModel
from .NTloss import NT_Xent

class RankModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.encoder_type = args.encoder_type
        if args.encoder_type == 'gru':
            self.embeddings = BertModel.from_pretrained(args.plm_path).embeddings
            self.encoder = nn.GRU(input_size=768,
                                    hidden_size=768,
                                    num_layers=1,
                                    batch_first=True,
                                    dropout=0.1,
                                    bidirectional=False)
        elif args.encoder_type == 'bert':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
            print(f'building bert-base model')
            self.encoder.resize_token_embeddings(30524)

        elif self.encoder_type == 'know':
            self.encoder = LukeModel.from_pretrained("studio-ousia/luke-base")
            
        self.pooler_type = args.pooler_type
        self.temperature = args.temperature if hasattr(args, 'temperature') else 1
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

        self.att_layer = nn.Sequential(
            nn.Linear(768, 300, bias=False),
            nn.Tanh(),
            nn.Linear(300, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)
    

    def pooler(self, hidden_states, attention_mask):
        att_logits = self.att_layer(hidden_states) - 1000 * (1-attention_mask.float().cuda()).unsqueeze(2)
        att_scores = self.softmax(att_logits).transpose(2, 1)
        pooler_output =  torch.bmm(att_scores, hidden_states).squeeze(1)
        return pooler_output



    def sentence_encoding(self, src_seq, src_mask):
        if self.encoder_type == 'gru':
            batch_size, max_src_len = src_seq.shape
            src_len = torch.sum(src_mask, 1)
            src_embedding = self.embeddings(src_seq)
            src_embedding = rnn.pack_padded_sequence(src_embedding, 
                                                     src_len, 
                                                     batch_first=True,
                                                     enforce_sorted=False)
            hidden_states, _ = self.encoder(src_embedding)
            hidden_states, _ = rnn.pad_packed_sequence(hidden_states, batch_first=True, total_length=max_src_len)

        elif self.encoder_type == 'bert':
            encoder_outputs = self.encoder(input_ids=src_seq, 
                                            attention_mask=src_mask,
                                            output_hidden_states=True)
            if isinstance(encoder_outputs, dict):
                hidden_states = encoder_outputs['hidden_states'][-1]
            else:
                hidden_states = encoder_outputs[-1][-1]

        elif self.encoder_type == 'know':
            encoder_outputs = self.encoder(input_ids=src_seq, 
                                            attention_mask=src_mask,
                                            output_hidden_states=True,
                                            return_dict = True)
            if isinstance(encoder_outputs, dict):
                hidden_states = encoder_outputs['hidden_states'][-1]
            else:
                hidden_states = encoder_outputs[-1][-1]
        src_pooler_output = self.pooler(hidden_states, src_mask)
        src_pooler_output = F.normalize(src_pooler_output, dim=-1)
        return src_pooler_output

    
    def forward(self, src_seq, src_mask, tgt_seq, tgt_mask):
        # cls:[bs, d_model]
        src_pooler_output = self.sentence_encoding(src_seq, src_mask)
        tgt_pooler_output = self.sentence_encoding(tgt_seq, tgt_mask)

        if self.training:
            # # [bs, bs]
            predict_logits = src_pooler_output.mm(tgt_pooler_output.t())
            predict_logits *= self.temperature
            # loss
            label = torch.arange(0, predict_logits.shape[0]).cuda()
            predict_loss = self.ce_loss(predict_logits, label)

            predict_result = torch.argmax(predict_logits, dim=1)
            acc = label == predict_result
            acc = (acc.int().sum() / (predict_logits.shape[0] * 1.0)).item()
          
            return predict_loss, acc
        else:
            return src_pooler_output, tgt_pooler_output





class RankModelKnow(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.encoder_type = args.encoder_type
        self.encoder = LukeModel.from_pretrained("studio-ousia/luke-base")
        print('building RankModelKnow model')
        self.pooler_type = args.pooler_type
        self.temperature = args.temperature if hasattr(args, 'temperature') else 1
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

        self.att_layer = nn.Sequential(
            nn.Linear(768, 300, bias=False),
            nn.Tanh(),
            nn.Linear(300, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)
    

    def pooler(self, hidden_states, attention_mask):
        att_logits = self.att_layer(hidden_states) - 1000 * (1-attention_mask.float().cuda()).unsqueeze(2)
        att_scores = self.softmax(att_logits).transpose(2, 1)
        pooler_output =  torch.bmm(att_scores, hidden_states).squeeze(1)
        return pooler_output


    def sentence_encoding(self, input_ids, entity_ids, entity_position_ids, attention_mask, entity_attention_mask):
        
        encoder_outputs = self.encoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        entity_ids=entity_ids,
                                        entity_attention_mask=entity_attention_mask,
                                        entity_position_ids=entity_position_ids,
                                        output_hidden_states=True,
                                        return_dict = True)
        if isinstance(encoder_outputs, dict):
            hidden_states = encoder_outputs['hidden_states'][-1]
            entity_hidden_states = encoder_outputs['entity_last_hidden_state']
        else:
            hidden_states = encoder_outputs[-1][-1]

        src_pooler_output = torch.cat((hidden_states, entity_hidden_states), dim=1)
        mask = torch.cat((attention_mask, entity_attention_mask), dim=1)
        src_pooler_output = self.pooler(src_pooler_output, mask)
        src_pooler_output = F.normalize(src_pooler_output, dim=-1)
        return src_pooler_output

    
    def forward(self, q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask, a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask):
        # cls:[bs, d_model]
        src_pooler_output = self.sentence_encoding(q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask)
        tgt_pooler_output = self.sentence_encoding(a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask)

        if self.training:
            # [bs, bs]
            predict_logits = src_pooler_output.mm(tgt_pooler_output.t())
            predict_logits *= self.temperature
            # loss
            label = torch.arange(0, predict_logits.shape[0]).cuda()
            predict_loss = self.ce_loss(predict_logits, label)

            predict_result = torch.argmax(predict_logits, dim=1)
            acc = label == predict_result
            acc = (acc.int().sum() / (predict_logits.shape[0] * 1.0)).item()


            return predict_loss, acc
        else:
            return src_pooler_output, tgt_pooler_output


class RankModelKnowledgePair(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.encoder_type = args.encoder_type
        self.encoder = LukeModel.from_pretrained("studio-ousia/luke-base")
        print('building RankModelKnowledgePair model')
        self.pooler_type = args.pooler_type
        self.temperature = args.temperature if hasattr(args, 'temperature') else 1
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

        self.att_layer = nn.Sequential(
            nn.Linear(768, 300, bias=False),
            nn.Tanh(),
            nn.Linear(300, 1, bias=False)
        )

        self.att_layer_for_entity = nn.Sequential(
            nn.Linear(768, 300, bias=False),
            nn.Tanh(),
            nn.Linear(300, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

        self.Qsk = nn.Linear(768, 128)
        self.Qss = nn.Linear(768, 128)
        self.Qks = nn.Linear(768, 128)
        self.Qkk = nn.Linear(768, 128)

        self.Ks = nn.Linear(768, 128)
        self.Kk = nn.Linear(768, 128)

        self.Vss = 1
        self.Vsk = 1
        self.Vks = 1
        self.Vkk = 1
    

    def pooler(self, hidden_states, attention_mask):
        att_logits = self.att_layer(hidden_states) - 1000 * (1-attention_mask.float().cuda()).unsqueeze(2)
        att_scores = self.softmax(att_logits).transpose(2, 1)
        pooler_output =  torch.bmm(att_scores, hidden_states).squeeze(1)
        return pooler_output

    def pooler_for_entity(self, hidden_states, attention_mask):
        att_logits = self.att_layer_for_entity(hidden_states) - 1000 * (1-attention_mask.float().cuda()).unsqueeze(2)
        att_scores = self.softmax(att_logits).transpose(2, 1)
        pooler_output = torch.bmm(att_scores, hidden_states).squeeze(1)
        return pooler_output

    def pairing(self, src_sentence_hidden_states, src_entity_hidden_states, tgt_sentence_hidden_states, tgt_entity_hidden_states, flag = False):
        s2k_query_layer = self.Qsk(src_sentence_hidden_states)
        s2s_query_layer = self.Qss(src_sentence_hidden_states)
        k2s_query_layer = self.Qks(src_entity_hidden_states)
        k2k_query_layer = self.Qkk(src_entity_hidden_states)

        s2k_key_layer = self.Kk(tgt_entity_hidden_states)
        s2s_key_layer = self.Kk(tgt_sentence_hidden_states)
        k2s_key_layer = self.Kk(tgt_sentence_hidden_states)
        k2k_key_layer = self.Kk(tgt_entity_hidden_states)

        s2k_attention_scores = torch.matmul(s2k_query_layer, s2k_key_layer.transpose(-1, -2))
        s2s_attention_scores = torch.matmul(s2s_query_layer, s2s_key_layer.transpose(-1, -2))
        k2s_attention_scores = torch.matmul(k2s_query_layer, k2s_key_layer.transpose(-1, -2))
        k2k_attention_scores = torch.matmul(k2k_query_layer, k2k_key_layer.transpose(-1, -2))
        
        score = s2k_attention_scores * self.Vsk + s2s_attention_scores * self.Vss + k2s_attention_scores * self.Vks + k2k_attention_scores * self.Vkk
        print(f'the shape of pairing matrix is {score.shape}')
        return score

    def sentence_encoding(self, input_ids, entity_ids, entity_position_ids, attention_mask, entity_attention_mask, flag = False):
        
        encoder_outputs = self.encoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        entity_ids=entity_ids,
                                        entity_attention_mask=entity_attention_mask,
                                        entity_position_ids=entity_position_ids,
                                        output_hidden_states=True,
                                        return_dict = True)
        if isinstance(encoder_outputs, dict):
            hidden_states = encoder_outputs['hidden_states'][-1]
            entity_hidden_states = encoder_outputs['entity_last_hidden_state']
        else:
            hidden_states = encoder_outputs[-1][-1]

        
        if flag == False:
            sentence_pooler_output = self.pooler(hidden_states, attention_mask)
            sentence_pooler_output = F.normalize(sentence_pooler_output, dim=-1)

            entity_pooler_output = self.pooler_for_entity(entity_hidden_states, entity_attention_mask)
            entity_pooler_output = F.normalize(entity_pooler_output, dim=-1)
            return sentence_pooler_output, entity_pooler_output
        
        elif flag == 'query':
            sentence_pooler_output = self.pooler(hidden_states, attention_mask)
            sentence_pooler_output = F.normalize(sentence_pooler_output, dim=-1)

            entity_pooler_output = self.pooler_for_entity(entity_hidden_states, entity_attention_mask)
            entity_pooler_output = F.normalize(entity_pooler_output, dim=-1)
            
            s2k_query_layer = self.Qsk(sentence_pooler_output)
            s2s_query_layer = self.Qss(sentence_pooler_output)
            k2s_query_layer = self.Qks(entity_pooler_output)
            k2k_query_layer = self.Qkk(entity_pooler_output)

            return s2k_query_layer, s2s_query_layer, k2s_query_layer, k2k_query_layer

        elif flag == 'key':
            sentence_pooler_output = self.pooler(hidden_states, attention_mask)
            sentence_pooler_output = F.normalize(sentence_pooler_output, dim=-1)

            entity_pooler_output = self.pooler_for_entity(entity_hidden_states, entity_attention_mask)
            entity_pooler_output = F.normalize(entity_pooler_output, dim=-1)
            k_key_layer = self.Kk(entity_pooler_output)
            s_key_layer = self.Kk(sentence_pooler_output)
            return k_key_layer, s_key_layer

    
    def forward(self, q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask, a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask, flag=False):
        # cls:[bs, d_model]
        src_sentence_hidden_states, src_entity_hidden_states = self.sentence_encoding(q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask, flag)
        tgt_sentence_hidden_states, tgt_entity_hidden_states = self.sentence_encoding(a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask, flag)

        if self.training:
            # [bs, bs]
            predict_logits = self.pairing(src_sentence_hidden_states, src_entity_hidden_states, tgt_sentence_hidden_states, tgt_entity_hidden_states)
            predict_logits *= self.temperature
            # loss
            label = torch.arange(0, predict_logits.shape[0]).cuda()
            predict_loss = self.ce_loss(predict_logits, label)

            predict_result = torch.argmax(predict_logits, dim=1)
            acc = label == predict_result
            acc = (acc.int().sum() / (predict_logits.shape[0] * 1.0)).item()


            return predict_loss, acc
        else:
            return src_sentence_hidden_states, src_entity_hidden_states, tgt_sentence_hidden_states, tgt_entity_hidden_states



class RankModelAddScore(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.encoder_type = args.encoder_type
        self.encoder = LukeModel.from_pretrained("studio-ousia/luke-base")
        print('building RankModelAddScore model')
        self.pooler_type = args.pooler_type
        self.temperature = args.temperature if hasattr(args, 'temperature') else 1
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

        self.att_layer = nn.Sequential(
            nn.Linear(768, 300, bias=False),
            nn.Tanh(),
            nn.Linear(300, 1, bias=False)
        )

        # self.att_layer_for_entity = nn.Sequential(
        #     nn.Linear(768, 300, bias=False),
        #     nn.Tanh(),
        #     nn.Linear(300, 1, bias=False)
        # )
        self.softmax = nn.Softmax(dim=1)

    
    def pooler(self, hidden_states, attention_mask):
        att_logits = self.att_layer(hidden_states) - 1000 * (1-attention_mask.float().cuda()).unsqueeze(2)
        att_scores = self.softmax(att_logits).transpose(2, 1)
        pooler_output =  torch.bmm(att_scores, hidden_states).squeeze(1)
        return pooler_output

    def pooler_for_entity(self, hidden_states, attention_mask):
        att_logits = self.att_layer(hidden_states) - 1000 * (1-attention_mask.float().cuda()).unsqueeze(2)
        att_scores = self.softmax(att_logits).transpose(2, 1)
        pooler_output = torch.bmm(att_scores, hidden_states).squeeze(1)
        return pooler_output

    def sentence_encoding(self, input_ids, entity_ids, entity_position_ids, attention_mask, entity_attention_mask, flag = False):
        
        encoder_outputs = self.encoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        entity_ids=entity_ids,
                                        entity_attention_mask=entity_attention_mask,
                                        entity_position_ids=entity_position_ids,
                                        output_hidden_states=True,
                                        return_dict = True)
        if isinstance(encoder_outputs, dict):
            hidden_states = encoder_outputs['hidden_states'][-1]
            entity_hidden_states = encoder_outputs['entity_last_hidden_state']
        else:
            hidden_states = encoder_outputs[-1][-1]

        
        sentence_pooler_output = self.pooler(hidden_states, attention_mask)
        sentence_pooler_output = F.normalize(sentence_pooler_output, dim=-1)

        entity_pooler_output = self.pooler_for_entity(entity_hidden_states, entity_attention_mask)
        entity_pooler_output = F.normalize(entity_pooler_output, dim=-1)
        return sentence_pooler_output, entity_pooler_output
    
    def forward(self, q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask, a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask, flag=False):
        # cls:[bs, d_model]
        src_sentence_hidden_states, src_entity_hidden_states = self.sentence_encoding(q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask, flag)
        tgt_sentence_hidden_states, tgt_entity_hidden_states = self.sentence_encoding(a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask, flag)

        if self.training:
            # [bs, bs]
            predict_logits = src_sentence_hidden_states.mm(tgt_sentence_hidden_states.t()) + src_entity_hidden_states.mm(tgt_entity_hidden_states.t())
            predict_logits *= self.temperature
            # loss
            label = torch.arange(0, predict_logits.shape[0]).cuda()
            predict_loss = self.ce_loss(predict_logits, label)

            predict_result = torch.argmax(predict_logits, dim=1)
            acc = label == predict_result
            acc = (acc.int().sum() / (predict_logits.shape[0] * 1.0)).item()

            # test new NT loss
            # batch_size = src_pooler_output.shape[0]
            # temperature = 1/self.temperature
            # NT_loss = NT_Xent(batch_size, temperature)
            # predict_loss = NT_loss(src_pooler_output, tgt_pooler_output)
            
            # # acc
            # predict_logits = src_pooler_output.mm(tgt_pooler_output.t())
            # label = torch.arange(0, predict_logits.shape[0]).cuda()
            # predict_result = torch.argmax(predict_logits, dim=1)
            # acc = label == predict_result
            # acc = (acc.int().sum() / (predict_logits.shape[0] * 1.0)).item()

            return predict_loss, acc
        else:
            return src_sentence_hidden_states, src_entity_hidden_states, tgt_sentence_hidden_states, tgt_entity_hidden_states

class RankModelGatedMatching(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.encoder_type = args.encoder_type
        self.encoder = LukeModel.from_pretrained("studio-ousia/luke-base")
        print('building RankModelGatedMatching model')
        self.pooler_type = args.pooler_type
        self.temperature = args.temperature if hasattr(args, 'temperature') else 1
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

        self.att_layer = nn.Sequential(
            nn.Linear(768, 300, bias=False),
            nn.Tanh(),
            nn.Linear(300, 1, bias=False)
        )

        # self.att_layer_for_entity = nn.Sequential(
        #     nn.Linear(768, 300, bias=False),
        #     nn.Tanh(),
        #     nn.Linear(300, 1, bias=False)
        # )

        self.knowledge_gate = nn.Linear(4, 1, bias = False)
        nn.init.normal_(self.knowledge_gate.weight.data, 0.5, 0.25) 

        self.softmax = nn.Softmax(dim=1)

    
    def pooler(self, hidden_states, attention_mask):
        att_logits = self.att_layer(hidden_states) - 1000 * (1-attention_mask.float().cuda()).unsqueeze(2)
        att_scores = self.softmax(att_logits).transpose(2, 1)
        pooler_output =  torch.bmm(att_scores, hidden_states).squeeze(1)
        return pooler_output

    def pooler_for_entity(self, hidden_states, attention_mask):
        att_logits = self.att_layer(hidden_states) - 1000 * (1-attention_mask.float().cuda()).unsqueeze(2)
        att_scores = self.softmax(att_logits).transpose(2, 1)
        pooler_output = torch.bmm(att_scores, hidden_states).squeeze(1)
        return pooler_output

    def score(self, Q, D):
        batch_size = Q.shape[0]
        return torch.einsum('bij, ckj->bcik', Q, D).reshape(batch_size,batch_size,4)


    def sentence_encoding(self, input_ids, entity_ids, entity_position_ids, attention_mask, entity_attention_mask, flag = False):
        
        encoder_outputs = self.encoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        entity_ids=entity_ids,
                                        entity_attention_mask=entity_attention_mask,
                                        entity_position_ids=entity_position_ids,
                                        output_hidden_states=True,
                                        return_dict = True)
        if isinstance(encoder_outputs, dict):
            hidden_states = encoder_outputs['hidden_states'][-1]
            entity_hidden_states = encoder_outputs['entity_last_hidden_state']
        else:
            hidden_states = encoder_outputs[-1][-1]

        
        sentence_pooler_output = self.pooler(hidden_states, attention_mask)
        sentence_pooler_output = F.normalize(sentence_pooler_output, dim=-1)

        entity_pooler_output = self.pooler_for_entity(entity_hidden_states, entity_attention_mask)
        entity_pooler_output = F.normalize(entity_pooler_output, dim=-1)

        pooler_output = torch.cat((sentence_pooler_output.unsqueeze(1), entity_pooler_output.unsqueeze(1)),dim=1)
        return pooler_output
    
    def forward(self, q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask, a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask, flag=False):
        # cls:[bs, d_model]
        src_pooler_output = self.sentence_encoding(q_input_ids, q_entity_ids, q_entity_position_ids, q_attention_mask, q_entity_attention_mask, flag)
        tgt_pooler_output = self.sentence_encoding(a_input_ids, a_entity_ids, a_entity_position_ids, a_attention_mask, a_entity_attention_mask, flag)
        print(f'self.gates weight is {self.knowledge_gate.weight}')
        if self.training:
            # [bs, bs]
            predict_logits = self.score(src_pooler_output, tgt_pooler_output)
            predict_logits = self.knowledge_gate(predict_logits).squeeze(2)
            predict_logits *= self.temperature
            # loss
            label = torch.arange(0, predict_logits.shape[0]).cuda()
            predict_loss = self.ce_loss(predict_logits, label)

            predict_result = torch.argmax(predict_logits, dim=1)
            acc = label == predict_result
            acc = (acc.int().sum() / (predict_logits.shape[0] * 1.0)).item()


            return predict_loss, acc
        else:
            return src_pooler_output, tgt_pooler_output