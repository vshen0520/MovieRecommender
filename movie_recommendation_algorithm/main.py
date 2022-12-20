import argparse
import os
import sys
import time
import json

import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

from NLP.config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from NLP.dataset_conv import CRSConvDataCollator, CRSConvDataset
from NLP.dataset_dbpedia import DBpedia
from NLP.evaluate_conv import ConvEvaluator
from NLP.model_gpt2 import PromptGPT2forCRS
from NLP.model_prompt import KGPrompt


from Recommendation.dataset_rec import CRSRecDataCollator, CRSRecDataset

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self
        

class CRSRec():
    def __init__(self):
        self.CRSprompt_encoder, self.CRStext_encoder, self.CRSmodel, self.CRStokenizer, self.CRStext_tokenizer, self.CRSkg, self.CRSargs = self.response()
        self.RECprompt_encoder, self.RECtext_encoder, self.RECmodel, self.RECtokenizer, self.RECtext_tokenizer, self.RECkg, self.RECargs = self.recommendation()
        
    def response(self):
        args = DotDict()
        args.dataset = 'inspired'
        args.fp16 = True
        args.split = 'test' 
        args.tokenizer = 'microsoft/DialoGPT-small'
        args.model = 'microsoft/DialoGPT-small'
        args.__module__text_tokenizer = 'roberta-base'
        args.text_encoder = 'roberta-base'
        args.conversation_encoder = '../../movie_recommendation_algorithm/NLP/pretrained_conversation/best'
        # args.conversation_encoder = './NLP/pretrained_conversation/best'
        args.seed = 2022
        args.n_prefix_conv = 20
        args.per_device_eval_batch_size = 1
        args.context_max_length = 200
        args.resp_max_length = 183
        args.prompt_max_length = 200
        args.entity_max_length = 32
        args.use_wandb = False
        args.debug = False
        args.text_tokenizer = 'roberta-base'
        args.num_bases = 8
        args.ignore_pad_token_for_loss= True
        args.num_workers = 8
        args.max_len = 200
        args.max_gen_len = 183

        # Initialize the accelerator. We will let the accelerator handle device placement for us.
        accelerator = Accelerator(device_placement=False, fp16=args.fp16)
        device = accelerator.device

        # Make one log on every process with the configuration for debugging.
        local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        logger.remove()
        logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
        logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
        logger.info(accelerator.state)
        # logger.info(config)

        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
        # wandb
        if args.use_wandb:
            name = args.name if args.name else local_time
            name += '_' + str(accelerator.process_index)

            if args.log_all:
                group = args.name if args.name else 'DDP_' + local_time
                run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
            else:
                if accelerator.is_local_main_process:
                    run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
                else:
                    run = None
        else:
            run = None

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenizer.add_special_tokens(gpt2_special_tokens_dict)
        model = PromptGPT2forCRS.from_pretrained(args.model)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model = model.to(device)

        text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
        text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
        text_encoder = AutoModel.from_pretrained(args.text_encoder)
        text_encoder.resize_token_embeddings(len(text_tokenizer))
        text_encoder = text_encoder.to(device)

        prompt_encoder = KGPrompt(
            model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
            n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
            edge_index=kg['edge_index'], edge_type=kg['edge_type'],
            n_prefix_rec=args.n_prefix_conv
        )
        if args.conversation_encoder is not None:
            prompt_encoder.load(args.conversation_encoder)
        prompt_encoder = prompt_encoder.to(device)
        prompt_encoder = accelerator.prepare(prompt_encoder)

        # model_name = args.conversation_encoder.split('/')[-2]
        return prompt_encoder, text_encoder, model, tokenizer, text_tokenizer, kg, args

    def recommendation(self):
        #args = parse_args()
        #config = vars(args)
        args = DotDict()

        args.dataset = 'inspired_gen'
        args.fp16 = True
        args.split = 'test' 
        args.tokenizer = 'microsoft/DialoGPT-small'
        args.model = 'microsoft/DialoGPT-small'
        args.text_tokenizer = 'roberta-base'
        args.text_encoder = 'roberta-base'
        args.recommendation_encoder = '../../movie_recommendation_algorithm/Recommendation/pretrained_recommendation_best/best'
        # args.recommendation_encoder = 'Recommendation/pretrained_recommendation_best/best'
        args.seed = 2022
        args.n_prefix_rec = 10
        args.per_device_eval_batch_size = 1
        args.context_max_length = 200
        args.resp_max_length = 183
        args.prompt_max_length = 200
        args.entity_max_length = 32
        args.use_wandb = False
        args.debug = False
        args.text_tokenizer = 'roberta-base'
        args.num_bases = 8
        args.ignore_pad_token_for_loss= True
        args.num_workers = 8
        args.max_len = 200
        args.max_gen_len = 183
        args.use_resp = False

        # Initialize the accelerator. We will let the accelerator handle device placement for us.
        accelerator = Accelerator(device_placement=False, fp16=args.fp16)
        device = accelerator.device

        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenizer.add_special_tokens(gpt2_special_tokens_dict)
        model = PromptGPT2forCRS.from_pretrained(args.model)
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model = model.to(device)

        text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
        text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
        text_encoder = AutoModel.from_pretrained(args.text_encoder)
        text_encoder.resize_token_embeddings(len(text_tokenizer))
        text_encoder = text_encoder.to(device)

        prompt_encoder = KGPrompt(
            model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
            n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
            edge_index=kg['edge_index'], edge_type=kg['edge_type'],
            n_prefix_rec=args.n_prefix_rec
        )
        if args.recommendation_encoder is not None:
            prompt_encoder.load(args.recommendation_encoder)
        prompt_encoder = prompt_encoder.to(device)

        return prompt_encoder, text_encoder, model, tokenizer, text_tokenizer, kg, args


    def conversation_recommend(self, dialog):
        print('------------------------------------------------------------------------------')
        print(dialog)

        f1 = open('../../movie_recommendation_algorithm/test_data_processed.jsonl', 'w', encoding='utf-8')
        # f1 = open('test_data_processed.jsonl', 'w', encoding='utf-8')
        raw = {'context':dialog, "resp": "", "rec": [], "entity": []}
        f1.write(json.dumps(raw, ensure_ascii=False) + '\n')
        f1.close()

        accelerator = Accelerator(device_placement=False, fp16=self.CRSargs.fp16)
        device = accelerator.device

        # data
        dataset = CRSConvDataset(
            self.CRSargs.dataset, self.CRSargs.split, self.CRStokenizer, debug=self.CRSargs.debug,
            context_max_length=self.CRSargs.context_max_length, resp_max_length=self.CRSargs.resp_max_length,
            entity_max_length=self.CRSargs.entity_max_length,
            prompt_tokenizer=self.CRStext_tokenizer, prompt_max_length=self.CRSargs.prompt_max_length
        )
        data_collator_generator = CRSConvDataCollator(
            tokenizer=self.CRStokenizer, device=device, gen=True, use_amp=accelerator.use_fp16, debug=self.CRSargs.debug,
            ignore_pad_token_for_loss=self.CRSargs.ignore_pad_token_for_loss,
            context_max_length=self.CRSargs.context_max_length, resp_max_length=self.CRSargs.resp_max_length,
            entity_max_length=self.CRSargs.entity_max_length, pad_entity_id=self.CRSkg['pad_entity_id'],
            prompt_tokenizer=self.CRStext_tokenizer
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.CRSargs.per_device_eval_batch_size,
            num_workers=self.CRSargs.num_workers,
            collate_fn=data_collator_generator,
        )

        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                token_embeds = self.CRStext_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds = self.CRSprompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=False,
                    use_conv_prefix=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds

                gen_seqs = accelerator.unwrap_model(self.CRSmodel).generate(
                    **batch['context'],
                    max_new_tokens=self.CRSargs.max_gen_len,
                    no_repeat_ngram_size=3,
                )
                #gen_seqs = tokenizer.batch_decode(gen_seqs, skip_special_tokens=True)
                gen_seqs = self.CRStokenizer.batch_decode(gen_seqs, skip_special_tokens=False)
                gen_seqs = gen_seqs[0].split('<|endoftext|>')[-2]
                gen_seqs = gen_seqs.split(':')[-1]

        f1 = open('../../movie_recommendation_algorithm/test_gen_data_processed.jsonl', 'w', encoding='utf-8')
        # f1 = open('test_gen_data_processed.jsonl', 'w', encoding='utf-8')
        raw = {'context':dialog, "resp": gen_seqs, "rec": [], "entity": []}
        f1.write(json.dumps(raw, ensure_ascii=False) + '\n')
        f1.close()

        fix_modules = [self.RECmodel, self.RECtext_encoder]
        for module in fix_modules:
            module.requires_grad_(False)

        # optim & amp
        modules = [self.RECprompt_encoder]
        test_dataset = CRSRecDataset(
            dataset=self.RECargs.dataset, split='test', debug=self.RECargs.debug,
            tokenizer=self.RECtokenizer, context_max_length=self.RECargs.context_max_length, use_resp=self.RECargs.use_resp,
            prompt_tokenizer=self.RECtext_tokenizer, prompt_max_length=self.RECargs.prompt_max_length,
            entity_max_length=self.RECargs.entity_max_length,
        )
        data_collator = CRSRecDataCollator(
            tokenizer=self.RECtokenizer, device=device, debug=self.RECargs.debug,
            context_max_length=self.RECargs.context_max_length, entity_max_length=self.RECargs.entity_max_length,
            pad_entity_id=self.RECkg['pad_entity_id'],
            prompt_tokenizer=self.RECtext_tokenizer, prompt_max_length=self.RECargs.prompt_max_length,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.RECargs.per_device_eval_batch_size,
            collate_fn=data_collator,
        )
        # test
        test_loss = []
        self.RECprompt_encoder.eval()
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                token_embeds = self.RECtext_encoder(**batch['prompt']).last_hidden_state
                prompt_embeds = self.RECprompt_encoder(
                    entity_ids=batch['entity'],
                    token_embeds=token_embeds,
                    output_entity=True,
                    use_rec_prefix=True
                )
                batch['context']['prompt_embeds'] = prompt_embeds
                batch['context']['entity_embeds'] = self.RECprompt_encoder.get_entity_embeds()

                outputs = self.RECmodel(**batch['context'], rec=True)
                test_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits[:, self.RECkg['item_ids']]
                
                #ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = torch.topk(logits, k=1, dim=-1).indices.tolist()
                ranks = [[self.RECkg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                labels = batch['context']['rec_labels']
                rec_item = ranks[0][0]

        with open('../../movie_recommendation_algorithm/id2entity.json', 'r', encoding='utf-8') as f:
        # with open('id2entity.json', 'r', encoding='utf-8') as f:
            id2entity = json.load(f)
        if '<movie>' in gen_seqs:
            gen_seqs = gen_seqs.replace('<movie>', id2entity[str(rec_item)])
        print('-------------------------')
        print(gen_seqs, rec_item)

        return gen_seqs

if __name__ == '__main__':
    # import re
    # with open('./Data/inspired/entity2id.json', 'r', encoding='utf-8') as f:
    #     entity2id = json.load(f)
    # id2entity = {}
    # for key,val in entity2id.items():
    #     id2entity[int(val)] = re.sub(r"[^a-zA-Z0-9()-?]+", ' ', key.split('/')[-1].split('>')[0]).strip()
    # with open('id2entity.json', 'w', encoding='utf-8') as f:
    #     json.dump(id2entity, f)

    # dialog = ["", "Hello there!", "hello", "What was the latest movie that you've watched?", "the last movie that i have watched is the irishman, which i loved"]
    # torch.multiprocessing.set_start_method('spawn')
    dialog = ["", "Hello, I hear that you are seeking a movie recommendation. Tell me about your movie preferences?", "I like action adventure and horror movies.", "Oh nice! Me too! I've recently watched The Ring, have you heard of it? It's really old, from 2002 I believe. It was a movie I watched as a kid.", "I heard of it and I think I watched it a while back but it was so long ago. I think they did remake a more recent version of it, did you watch the remake of it?", "No, I don't believe I've seen a remake. I think there was a sequel that came out a while back, but that's as far as I've gone with this. What recent movies have you seen? Have you seen Zombieland? Not sure if that's a horror movie, though. Sure sounds like it.", "I recently watched the new Halloween with michael myers and Anabelle. I haven't seen Zombieland but I think its a comedy if im correct. What horror movies would you suggest watching that came out in 2019?"]
    # dialog = ["", "I like action adventure and horror movies."]
    MovieREC = CRSRec()
    MovieREC.conversation_recommend(dialog)
    # conversation_recommend(dialog)
