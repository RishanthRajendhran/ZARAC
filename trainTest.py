import argparse
import wandb
import logging
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, AdamW, get_scheduler, T5ForConditionalGeneration, T5Tokenizer
from torch.optim import Adam
from datasets import Dataset, load_dataset
import numpy as np
import pandas as pd
import json
import deepspeed
import json
from tqdm import tqdm
import regex as re
import os
import glob
from os.path import exists
from pathlib import Path
from torch.utils.data import DataLoader
import bitsandbytes as bnb
import math
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
import math
from transformers import pipeline
import gc

# TRACK_TRAINING=True
# TRACK_INFERENCE=True
# TRACK_NLI=True

TRACK_TRAINING=False
TRACK_INFERENCE=False
TRACK_NLI=False

MODEL_TEMPERATURE=0.1
MODEL_TOP_P=0.9
MODEL_TOP_K=0
MODEL_MAX_NEW_TOKENS=128
MODEL_DO_SAMPLE=True
MODEL_REPETITION_PENALTY=1.0
# MINLENGTH=32
MAXLENGTH=512
DS_MAX_LENGTH=4096
NLIlabels = {
    "Entailment":{
        "ecqa": {
            "threshold":  0.55
            # "threshold":  0.9
        },
    },
    "Neutral":{}, 
    "Contradiction":{},
}
TargetNLIlabels = {
    "ecqa": "Entailment",
}

supportedModels = ["unifiedqa"]
supportedSizes = {
    "unifiedqa": ["3b"],
}
supportedDatasets = ["ecqa"]
supportedHFDatasets = []

parser = argparse.ArgumentParser()

parser.add_argument(
    "-log",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-episodes",
    type=int,
    help="No. of episodes",
    # default=60
    default=30
)

parser.add_argument(
    "-startEpisode",
    type=int,
    help="Episode to start from",
    default=0
)

parser.add_argument(
    "-endEpisode",
    type=int,
    help="Episode after the episode to end at",
    default=-1
)

parser.add_argument(
    "-dataPrefix",
    help="Prefix to path to data (train/test) files for each episode [Expected format: '/path/to/folder/containing/episodes/episode_<episodeNumber>' where ''/path/to/folder/containing/episodes/' is the prefix]",
    default="/uufs/chpc.utah.edu/common/home/u1419542/ZARA/datasets/ecqa/episodes/"
)

parser.add_argument(
    "-trainName",
    help="Train files name with extension",
    default="train.jsonl"
)

parser.add_argument(
    "-testName",
    help="Test files name with extension",
    default="test.jsonl"
)

parser.add_argument(
    "-model",
    choices=supportedModels,
    help="Name of HuggingFace model to use",
    default="unifiedqa"
)

parser.add_argument(
    "-size",
    help="Size of HuggingFace model to use",
    default="3b"
)

parser.add_argument(
    "-dataset",
    choices=supportedDatasets,
    help="Name of dataset to use",
    default="ecqa"
)

parser.add_argument(
    "-peft",
    action="store_true",
    help="Boolean flag to perform PEFT"
)

parser.add_argument(
    "-trainPrompt",
    help="Path to file containing few-shot prompt to include with every train instance",
    default="None"  
)

parser.add_argument(
    "-testPrompt",
    help="Path to file containing few-shot prompt to include with every test instance",
    default="None"  
)

parser.add_argument(
    "-maxShots",
    type=int,
    help="Maximum no. of shots to use in few-shot setting",
    default=9
)

parser.add_argument(
    "-direct",
    action="store_true",
    help="Boolean flag to enable direct prompting"
)

parser.add_argument(
    "-numEpochs",
    type=int,
    help="No. of epochs to finetune for",
    default=1
)

parser.add_argument(
    "-maxSteps",
    type=int,
    help="Maximum number of optimization steps allowed",
    default=1
)

parser.add_argument(
    "-batchSize",
    type=int,
    help="Training Batchsize",
    default=8
)

parser.add_argument(
    "-learningRate",
    type=float,
    help="Learning rate for training",
    default=3e-5,
)

parser.add_argument(
    "-savePath",
    type=str,
    help="Path to save model after every epoch of finetuning",
    default="./model/"
)

parser.add_argument(
    "-NLImodels",
    nargs="+",
    help="List of NLI models to use in the plausibility approximator ensemble",
    default=["roberta-large-mnli", "microsoft/deberta-large-mnli", "facebook/bart-large-mnli"]
)

parser.add_argument(
    "-deepSpeedTrain",
    action="store_true",
    help="Boolean flag to indicate training through deepspeed"
)

parser.add_argument(
    "-deepSpeedTest",
    action="store_true",
    help="Boolean flag to indicate testing through deepspeed"
)

parser.add_argument(
    "-lrScheduler",
    type=int,
    help="When set, specifies the no. of warmup steps to perform in a linear learning rate schedule [Default: -1, no learning rate scheduler used]",
    default=-1
)

#Arguments for DeepSpeed
parser.add_argument(
    "--local_rank", 
    type=int, 
    help="[DEEPSPEED ARGUMENT]",
    default=0
)

parser.add_argument(
    "--do_eval",
    action="store_true",
    help="[DEEPSPEED ARGUMENT] Boolean flag to enable inference mode"
)

parser.add_argument(
    "--deepspeed", 
    help="[DEEPSPEED ARGUMENT] Path to deepspeed configuration"
)

#---------------------------------------------------------------------------
class DatasetTokenizer():
    def __init__(self, modelName, tokenizer, dataset, direct=False, trainPrompt="s"):
        self.modelName = modelName
        self.tokenizer = tokenizer
        if dataset not in supportedDatasets:
            raise ValueError(f"{dataset} not supported!")
        self.dataset = dataset
        self.direct = direct
        self.trainPrompt = trainPrompt

    def _generateIndividualPrompt(self, instance): 
        if self.modelName == "gptj":
            #commonsense_qa on HuggingFace
            # {
            #     "id": (string),
            #     "question": (string),
            #     "choices": {
            #         "labels": [(string),...],
            #         "text": [(string),...]
            #     },
            #     "rationale": (string),
            #     "answerKey": (string)
            # }
            if self.dataset == "commonsense_qa":
                prompt = self.trainPrompt
                prompt += "Q: " + instance["question"] + "\nAnswer Choices:\n"
                corrAns = ""
                for c, t in zip(instance["choices"]["label"], instance["choices"]["text"]):
                    prompt += "({}) {}".format(c.lower(), t.lower())
                    prompt += "\n"
                    if c.lower() == instance["answerKey"].lower():
                        corrAns = t
                prompt += "A: "
                if self.direct: 
                    prompt += "({}).\n\n".format(instance["answerKey"].lower())
                else:
                    prompt += "{} Therefore, the answer is {} ({}).\n\n".format(instance["rationale"], corrAns.lower(), instance["answerKey"].lower())
            #ecqa
            # {
            #     "id": (string),
            #     "question": {
            #        "question_concept": (string),
            #        "choices": [{
            #            "labels": (string),
            #            "text": (string)
            #        },...],
            #       "stem": (string)
            #     },
            #     "positives": [(string),...],
            #     "negatives": [(string),...],
            #     "explanation": (string),
            #     "answerKey": (string)
            # }
            elif self.dataset == "ecqa":
                prompt = self.trainPrompt
                prompt += "explain " + instance["question"]["stem"].lower() + " \\n"
                corrAns = ""
                for ind in range(len(instance["question"]["choices"])):
                    c = instance["question"]["choices"][ind]["label"]
                    t = instance["question"]["choices"][ind]["text"]
                    prompt += " ({}) {}".format(c.lower(), t.lower())
                    if c.lower() == instance["answerKey"].lower():
                        corrAns = t
                if self.direct: 
                    prompt += "({})".format(instance["answerKey"].lower())
                else:
                    explanation = '. '.join(instance['positives'])
                    explanation += '.'
                    explanation = explanation.replace('..','.')
                    prompt += "({}) {} because {}".format(instance["answerKey"].lower(), corrAns.lower(), explanation)
            #gsm8k on HuggingFace
            # {
            #     "question": (string),
            #     "answer": (string)
            # }
            elif self.dataset == "gsm8k": 
                prompt = self.trainPrompt
                prompt += "Q: " + instance["question"] 
                extractedAnswer = extractAnswer(instance["answer"], self.dataset, self.direct)
                prompt += "\nA: "
                if self.direct: 
                    prompt += extractedAnswer["answer"]
                else:
                    prompt += instance["answer"]
            else: 
                raise NotImplementedError(f"Prompt generation not yet implemented for {self.dataset}")
            return prompt
        elif self.modelName == "unifiedqa":
            if self.dataset == "commonsense_qa":
                prompt = self.trainPrompt
                label = ""
                prompt += "Q: " + instance["question"] + "\nAnswer Choices:\n"
                corrAns = ""
                for c, t in zip(instance["choices"]["label"], instance["choices"]["text"]):
                    prompt += "({}) {}".format(c.lower(), t.lower())
                    prompt += "\n"
                    if c.lower() == instance["answerKey"].lower():
                        corrAns = t
                prompt += "A: "
                if self.direct: 
                    label += "({}).\n\n".format(instance["answerKey"].lower())
                else:
                    label += "{} Therefore, the answer is {} ({}).\n\n".format(instance["rationale"], corrAns.lower(), instance["answerKey"].lower())
            # elif self.dataset == "ecqa":
            #     prompt = self.trainPrompt
            #     label = ""
            #     prompt += "Q: " + instance["question"]["stem"] + "\nAnswer Choices:\n"
            #     corrAns = ""
            #     for ind in range(len(instance["question"]["choices"])):
            #         c = instance["question"]["choices"][ind]["label"]
            #         t = instance["question"]["choices"][ind]["text"]
            #         prompt += "({}) {}".format(c.lower(), t.lower())
            #         prompt += "\n"
            #         if c.lower() == instance["answerKey"].lower():
            #             corrAns = t
            #     prompt += "A: "
            #     if self.direct: 
            #         label += "({}).\n\n".format(instance["answerKey"].lower())
            #     else:
            #         explanation = '. '.join(instance['positives'])
            #         explanation += '.'
            #         explanation = explanation.replace('..','.')
            #         label += "{} Therefore, the answer is {} ({}).\n\n".format(explanation, corrAns.lower(), instance["answerKey"].lower())
            elif self.dataset == "ecqa":
                prompt = self.trainPrompt
                prompt += "explain " + instance["question"]["stem"].lower() + " \\n"
                label = ""
                corrAns = ""
                for ind in range(len(instance["question"]["choices"])):
                    c = instance["question"]["choices"][ind]["label"]
                    t = instance["question"]["choices"][ind]["text"]
                    prompt += " ({}) {}".format(c.lower(), t.lower())
                    if c.lower() == instance["answerKey"].lower():
                        corrAns = t
                if self.direct: 
                    label += "({})</s>".format(instance["answerKey"].lower())
                else:
                    explanation = '. '.join(instance['positives'])
                    explanation += '.'
                    explanation = explanation.replace('..','.')
                    label += "({}) {} because {}</s>".format(instance["answerKey"].lower(), corrAns.lower(), explanation)
                prompt = prompt.lower()
                label = label.lower()
            elif self.dataset == "gsm8k": 
                prompt = self.trainPrompt
                label = ""
                prompt += "Q: " + instance["question"] 
                extractedAnswer = extractAnswer(instance["answer"], self.dataset, self.direct)
                prompt += "\nA: "
                if self.direct: 
                    label += extractedAnswer["answer"]
                else:
                    label += instance["answer"]
            else: 
                raise NotImplementedError(f"Prompt generation not yet implemented for {self.dataset}")
            return prompt, label
        else: 
            raise ValueError("{} model not supported!".format(self.modelName))

    def tokenize(self, instances):
        if self.modelName == "gptj":
            prompt = self._generateIndividualPrompt(instances)
            tokenizedInput = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=MAXLENGTH)
            return tokenizedInput
        elif self.modelName == "unifiedqa":
            prompt, label = self._generateIndividualPrompt(instances)
            tokenizedInput = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=MAXLENGTH)
            tokenizedLabels = self.tokenizer(label, return_tensors="pt", truncation=True, padding="max_length", max_length=MAXLENGTH).input_ids
            tokenizedInput.update({
                "labels": torch.squeeze(tokenizedLabels),
                "input_ids": torch.squeeze(tokenizedInput.input_ids),
                "attention_mask": torch.squeeze(tokenizedInput.attention_mask),
            })
            return tokenizedInput
        else: 
            raise ValueError("{} model not supported!".format(self.modelName))
#---------------------------------------------------------------------------
def _generateIndividualPrompt(instance, dataset, model, direct=False, rationalize=False, isTest=False):
    if dataset not in supportedDatasets:
        raise ValueError(f"{dataset} not supported!")
    
    prompt = ""
    
    #commonsense_qa on HuggingFace
    # {
    #     "id": (string),
    #     "question": (string),
    #     "choices": {
    #         "labels": [(string),...],
    #         "text": [(string),...]
    #     },
    #     "answerKey": (string)
    # }
    if dataset == "commonsense_qa":
        if not direct and not isTest: 
            raise ValueError("Only direct prompting supported with commonsense_qa dataset on HuggingFace!")
        prompt += "Q: " + instance["question"] + "\nAnswer Choices:\n"
        for c, t in zip(instance["choices"]["label"], instance["choices"]["text"]):
            prompt += "({}) {}".format(c.lower(), t.lower())
            if rationalize:
                if c == instance["answerKey"]:
                    prompt += " (CORRECT)" 
            prompt += "\n"
        prompt += "A: "
        if not isTest: 
            prompt += "({}).\n\n".format(instance["answerKey"].lower())
    #ecqa
    # {
    #     "id": (string),
    #     "question": {
    #        "question_concept": (string),
    #        "choices": [{
    #            "labels": (string),
    #            "text": (string)
    #        },...],
    #       "stem": (string)
    #     },
    #     "positives": [(string),...],
    #     "negatives": [(string),...],
    #     "explanation": (string),
    #     "answerKey": (string)
    # }
    elif dataset == "ecqa":
        prompt += "explain " + instance["question"]["stem"].lower() + "\\n"
        corrAns = ""
        for ind in range(len(instance["question"]["choices"])):
            c = instance["question"]["choices"][ind]["label"]
            t = instance["question"]["choices"][ind]["text"]
            prompt += " ({}) {}".format(c.lower(), t.lower())
            if c.lower() == instance["answerKey"].lower():
                corrAns = t
        if not isTest:
            if direct: 
                prompt += "({})".format(instance["answerKey"].lower())
            else:
                explanation = '. '.join(instance['positives'])
                explanation += '.'
                explanation = explanation.replace('..','.')
                prompt += "({}) {} because {}".format(instance["answerKey"].lower(), corrAns.lower(), explanation)
        if model == "unifiedqa":
            prompt = prompt.lower()
        # prompt += "Q: " + instance["question"]["stem"] + "\nAnswer Choices:\n"
        # for ind in range(len(instance["question"]["choices"])):
        #     c = instance["question"]["choices"][ind]["label"]
        #     t = instance["question"]["choices"][ind]["text"]
        #     prompt += "({}) {}".format(c.lower(), t.lower())
        #     if rationalize:
        #         if c == instance["answerKey"]:
        #             prompt += " (CORRECT)" 
        #     prompt += "\n"
        # prompt += "A: "
        # if not isTest: 
        #     if direct: 
        #         prompt += "({}).\n\n".format(instance["answerKey"].lower())
        #     else:
        #         explanation = '. '.join(instance['positives'])
        #         explanation += '.'
        #         explanation = explanation.replace('..','.')
        #         prompt += "{} Therefore, the answer is {} ({}).\n\n".format(explanation, corrAns.lower(), instance["answerKey"].lower())
    #gsm8k on HuggingFace
    # {
    #     "question": (string),
    #     "answer": (string)
    # }
    elif dataset == "gsm8k": 
        prompt += "Q: " + instance["question"] 
        extractedAnswer = extractAnswer(instance["answer"], dataset, direct)
        if rationalize:
            prompt += " ({})".format(extractedAnswer["answer"]) 
        prompt += "\nA: "
        if not isTest: 
            if direct:
                prompt += extractedAnswer["answer"] + "\n\n"
            else:
                prompt += instance["answer"] + "\n\n"
    #arithmetic (Not on Huggingface)
    # {
    #     "question": (string),
    #     "answer": (string),
    #     "scratch": (string), [OPTIONAL]
    # }
    elif dataset == "arithmetic": 
        prompt += "Input:\n" + instance["question"].strip() 
        prompt += "\nTarget:\n"
        if rationalize:
            prompt += instance["answer"] + "\n\n"
        if not isTest: 
            if not direct: 
                prompt += instance["scratch"].strip() + "\n"
            prompt += instance["answer"] + "\n\n"
    return prompt

#---------------------------------------------------------------------------
def _generatePrompt(data, dataset, model, maxShots, direct=False, rationalize=False, isTest=False):
    prompts = []

    if dataset not in supportedDatasets:
        raise ValueError(f"{dataset} not supported!")

    for index, instance in enumerate(data):
        if index >= maxShots:
            break 
        prompts.append(_generateIndividualPrompt(instance, dataset, model, direct, rationalize, isTest))
    
    return prompts
#---------------------------------------------------------------------------
def generateTrainPrompt(data, dataset, model, maxShots, direct, rationalize=False):
    return "".join(_generatePrompt(data, dataset, model, maxShots, direct, rationalize, False))
#---------------------------------------------------------------------------
def generateTestPrompt(instance, dataset, model, maxShots, direct, rationalize=False):
    return "".join(_generatePrompt([instance], dataset, model, maxShots, direct, rationalize, True))
#---------------------------------------------------------------------------
def extractAnswer(answer, dataset, direct=False):
    if dataset not in supportedDatasets:
        raise ValueError(f"{dataset} not supported!")
    if dataset == "commonsense_qa":
        if not direct:
            searchPattern = "answer is .*."
        else: 
            searchPattern = "\([a-z]\)."
        matchedSpan = re.search(searchPattern, answer)
        if matchedSpan == None:
            logging.warning(f"Could not extract answer from {answer}!")
            return None
            # raise RuntimeError(f"Could not extract answer from {answer}!")
        extractedAnswer = answer[matchedSpan.start():matchedSpan.end()].strip()
        answerPattern = "\([a-z]\)."
        matchedAnswer = re.findall(answerPattern, extractedAnswer)
        if len(matchedAnswer)==0:
            logging.warning(f"Could not extract answer from {extractedAnswer}!")
            return None
            # raise RuntimeError(f"Could not extract answer from {extractedAnswer}!")
        matchedAnswer = matchedAnswer[-1][1]
        extractedAnswer = {
            "answer":matchedAnswer.strip(),
        }
        if not direct:
            rationale = answer[:matchedSpan.start()]
            rationalePattern = "[.]"
            matchedRationale = re.split(rationalePattern, rationale)
            if len(matchedRationale):
                rationale = ".".join(matchedRationale[:-1])+"."
            extractedAnswer.update({
                "rationale":rationale.strip(), 
            })
    elif dataset == "ecqa":
        if not direct:
            searchPattern = "\([a-z]\) .* because"
        else: 
            searchPattern = "\([a-z]\)."
        matchedSpan = re.search(searchPattern, answer)
        if matchedSpan == None:
            logging.warning(f"Could not extract answer from {answer}!")
            return None
            # raise RuntimeError(f"Could not extract answer from {answer}!")
        extractedAnswer = answer[matchedSpan.start():matchedSpan.end()].strip()
        answerPattern = "\([a-z]\)"
        matchedAnswer = re.findall(answerPattern, extractedAnswer)
        if len(matchedAnswer)==0:
            logging.warning(f"Could not extract answer from {extractedAnswer}!")
            return None
            # raise RuntimeError(f"Could not extract answer from {extractedAnswer}!")
        answerString = extractedAnswer[extractedAnswer.index(matchedAnswer[-1])+len(matchedAnswer[-1]):]
        if not direct and "because" in answerString:
            answerString = answerString[:answerString.index("because")]
        matchedAnswer = matchedAnswer[-1][1]
        extractedAnswer = {
            "answer":matchedAnswer.strip(),
            "answerString":answerString.strip(),
        }
        if not direct:
            rationale = answer[matchedSpan.end():]
            rationale = re.sub("</s>", "", rationale)
            extractedAnswer.update({
                "rationale":rationale.strip(), 
            })
    elif dataset == "gsm8k":
        if not direct:
            searchPattern = "\n#### [0-9]+"
        else: 
            searchPattern = "[0-9]+"
        matchedSpan = re.search(searchPattern, answer)
        if matchedSpan == None:
            logging.warning(f"Could not extract answer from {answer}!")
            return None
            # raise RuntimeError(f"Could not extract answer from {answer}!")
        extractedAnswer = answer[matchedSpan.start():matchedSpan.end()].strip()
        if not direct:
            matchedAnswer = re.sub("#","",extractedAnswer).strip()
        else:
            matchedAnswer = extractedAnswer.strip()
        extractedAnswer = {
            "answer":matchedAnswer.strip(),
        }
        if not direct:
            rationale = answer[:matchedSpan.start()]
            extractedAnswer.update({
                "rationale":rationale.strip(), 
            })
    elif dataset == "arithmetic":
        if not direct:
            searchPattern = "</scratch>\n"
        else: 
            searchPattern = "Target:\n"
        matchedSpan = re.search(searchPattern, answer)
        if matchedSpan == None:
            logging.warning(f"Could not extract answer from {answer}!")
            return None
            # raise RuntimeError(f"Could not extract answer from {answer}!")
        matchedAnswer = answer[matchedSpan.end():].strip()
        if "\n" in matchedAnswer:
            matchedAnswer = matchedAnswer[:matchedAnswer.index("\n")]
        extractedAnswer = {
            "answer":matchedAnswer.strip(),
        }
        if not direct:
            scratchStart = "<scratch>"
            scratchEnd = "</scratch>\n"
            matchedStartSpan = re.search(scratchStart, answer)
            matchedEndSpan = re.search(scratchEnd, answer)
            scratch = answer[matchedStartSpan.start():matchedEndSpan.end()]
            extractedAnswer.update({
                "rationale":scratch.strip(), 
            })
    return extractedAnswer
#---------------------------------------------------------------------------
def _generateIndividualNLIPrompt(instance, dataset, model, isTest=False):
    if dataset not in supportedDatasets:
        raise ValueError(f"{dataset} not supported!")
    
    prompt = ""
    
    #ecqa
    # {
    #     "id": (string),
    #     "question": {
    #        "question_concept": (string),
    #        "choices": [{
    #            "labels": (string),
    #            "text": (string)
    #        },...],
    #       "stem": (string)
    #     },
    #     "positives": [(string),...],
    #     "negatives": [(string),...],
    #     "explanation": (string),
    #     "answerKey": (string)
    # }
    if dataset == "ecqa":
        NLIprompt= {
            "default": "Premise: Because {premise}\nHypothesis: The answer of the question \"{question}\" is {answer}.\nNLI Class: {NLIclass}",
            "isTest": "Premise: Because {premise}\nHypothesis: The answer of the question \"{question}\" is {answer}.\nNLI Class: ",
        }

        if isTest:
            prompt += NLIprompt["isTest"].format(premise=instance["explanation"], question=instance["question"]["stem"], answer=instance["answerString"])
        else: 
            prompt += NLIprompt["default"].format(premise=instance["explanation"], question=instance["question"]["stem"], answer=instance["answerString"], NLIclass=TargetNLIlabels[dataset])
    else:
        raise ValueError("NLI mapping for {} dataset not supported!")
    return prompt

#---------------------------------------------------------------------------
def mapToNLI(data, dataset, model, maxShots, isTest=False):
    NLIprompts = []

    if dataset not in supportedDatasets:
        raise ValueError(f"{dataset} not supported!")

    for index, instance in enumerate(data):
        if index >= maxShots:
            break 
        NLIprompts.append(_generateIndividualNLIPrompt(instance, dataset, model, isTest))
    
    return NLIprompts
#---------------------------------------------------------------------------
def getTrainDataLoader(trainData, dataset, model, tokenizer, batchSize, direct=False, useTrainPrompt="None"):
    trainDS = Dataset.from_list(trainData)

    trainPrompt=""
    if useTrainPrompt!="None":
        if useTrainPrompt.endswith(".txt"):
            with open(useTrainPrompt, "r") as f:
                trainPrompt = f.read()
        else:
            raise ValueError("Invalid train prompt: {}".format(useTrainPrompt))

    if dataset == "ecqa":
        tokenizedTrainDS = trainDS.map(DatasetTokenizer(model, tokenizer, dataset, direct, trainPrompt).tokenize, batched=False, remove_columns=trainDS.column_names)
    else: 
        raise NotImplementedError("Support for {} not yet implemented!".format(dataset))

    if model == "unifiedqa":
        dataCollator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    else: 
        raise NotImplementedError("Data collator for {} not specified!".format(model))
    
    tokenizedTrainDS.set_format("torch")
    trainDataLoader = DataLoader(
        tokenizedTrainDS, 
        shuffle=True, 
        batch_size=batchSize, 
        collate_fn=dataCollator
    )

    return trainDataLoader, dataCollator
#---------------------------------------------------------------------------
def loadModel(model, size, device, train=False, peft=False, modelID=None, deepSpeed=False, localRank=None):
    dsModel = None
    if model == "unifiedqa":
        if not modelID:
            if size == "3b":
                # modelID = "allenai/unifiedqa-t5-3b"
                modelID = "allenai/unifiedqa-v2-t5-3b-1251000"
            elif size == "large":
                # modelID = "allenai/unifiedqa-t5-large"
                modelID = "allenai/unifiedqa-v2-t5-large-1251000"
            elif size == "base":
                # modelID = "allenai/unifiedqa-t5-base"
                modelID = "allenai/unifiedqa-v2-t5-base-1251000"
            else: 
                raise ValueError("Only {} size(s) supported!".format("/".join(supportedSizes[model])))
        
        logging.info("Using pretrained model and tokenizer from {} on HuggingFace".format(modelID))
        model = T5ForConditionalGeneration.from_pretrained(modelID, device_map="auto")
        if train:
            model.gradient_checkpointing_enable()
            if peft:
                logging.info("Using PEFT for finetuning")
                loraConfig = LoraConfig(
                    r=8, 
                    lora_alpha=32, 
                    lora_dropout=0.05, 
                    bias="none", 
                    task_type="SEQ_2_SEQ_LM"
                )
                model = prepare_model_for_int8_training(model)
                model = get_peft_model(model, loraConfig)
        for p in model.parameters():
            p = p.contiguous()
        tokenizer = T5Tokenizer.from_pretrained(modelID)
        tokenizer.pad_token = tokenizer.eos_token
    else: 
        raise ValueError("Only {} model(s) supported!".format("/".join(supportedModels)))
    
    if train:
        model.train()
        model.to(device=device)
    else:
        model.eval()
        if deepSpeed:
            if model.dtype == torch.float32:
                model.to(device=localRank)    
            dsModel = deepspeed.init_inference(
                model=model,
                # dtype=torch.half,
                mp_size=1,
                replace_with_kernel_inject=True,
                max_out_tokens=DS_MAX_LENGTH,
            )
            # assert isinstance(dsModel.module.transformer.h[0], DeepSpeedTransformerInference) == True, "Model not sucessfully initalized"
        else: 
            if model.dtype == torch.float32:
                model.to(device=device)

    return model, tokenizer, modelID, dsModel
#---------------------------------------------------------------------------
def setupTrainer(model, learningRate, useLrScheduler=-1, numTrainingSteps=-1, deepSpeed=False, dataCollator=None, deepSpeedConfig=None):
    # optimizer = bnb.optim.Adam8bit(model.parameters(), lr=learningRate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    if useLrScheduler > -1:
        if numTrainingSteps == -1:
            raise ValueError("numTrainingSteps should be passed to setupTrainer function when useLrScheduler is set!")
        lrScheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=useLrScheduler,
            num_training_steps=numTrainingSteps,
        )
    else:
        lrScheduler=None

    if deepSpeed:
        modelEngine, modelOptimizer, _, _ = deepspeed.initialize(
            args={
                "zero_allow_untested_optimizer": True,
            },
            model=model,
            model_parameters=model.parameters(),
            optimizer=optimizer,
            lr_scheduler=lrScheduler,
            collate_fn=dataCollator,
            config=deepSpeedConfig,
        )
    else: 
        modelEngine, modelOptimizer = None, None

    return optimizer, lrScheduler, modelEngine, modelOptimizer
#---------------------------------------------------------------------------
def performTraining(model, saveModelPath, device, optimizer, trainDataLoader, numEpochs, maxSteps, useLrScheduler=False, lrScheduler=None, trackTraining=False, tokenizer=None, deepSpeedTrain=False, modelEngine=None):
    if trackTraining and not tokenizer:
        raise RuntimeError("Tokenizer should be provided to train function when trackTraining is set!")
    
    numTrainingSteps = numEpochs * len(trainDataLoader)
    progressBar = tqdm(range(numTrainingSteps))
    bestLoss = np.inf
    numSteps = 0
    for epoch in tqdm(range(numEpochs),desc="Epoch"):
        model.train()
        batchInd = 0
        avgLoss = 0
        for batch in tqdm(trainDataLoader, desc="Batch"):
            numSteps += 1
            batchInd += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            if deepSpeedTrain:
                loss = modelEngine(batch)
                modelEngine.backward(loss)
                modelEngine.step()
            else:
                outputs = model(**batch)
                loss = outputs.loss 

                if trackTraining:
                    logits = torch.squeeze(outputs.logits)
                    outputIDs = logits.argmax(-1)
                    genText = tokenizer.batch_decode(outputIDs)
                    inpText = tokenizer.batch_decode(torch.squeeze(batch["input_ids"], dim=1))
                    labels = tokenizer.batch_decode(torch.squeeze(batch["labels"], dim=1))
                    for inp, out, label in zip(inpText, genText, labels):
                        logging.info("Epoch {}/{}:".format(epoch, numEpochs))
                        logging.info(f"Input:\n{inp}")
                        logging.info(f"Label:\n{label}")
                        logging.info(f"Output:\n{out}")
                        logging.info("-"*25) 

                loss.backward()
                optimizer.step()
                if useLrScheduler > -1:
                    lrScheduler.step()  
                optimizer.zero_grad()
            avgLoss += loss.item()
            progressBar.update(1)
            if numSteps >= maxSteps:
                break
        #Update
        avgLoss /= batchInd
        logging.info(f"Epoch {epoch+1}/{numEpochs}, Step {numSteps}/{maxSteps}: Loss = {avgLoss}")
        wandb.log({
            "stage_1_train_loss": avgLoss
        })
        if avgLoss < bestLoss:
            bestLoss = avgLoss
            model.save_pretrained(saveModelPath, from_pt=True) 
            tokenizer.save_pretrained(saveModelPath, from_pt=True)
        if numSteps >= maxSteps:
            break
#---------------------------------------------------------------------------
def infer(model, modelName, tokenizer, prompt, generationConfig={}, deepSpeed=False, dsModel=None):
    tokenizedInput = tokenizer(prompt, return_tensors="pt")
    inputIDs = tokenizedInput.input_ids.to(device=model.device)
    attentionMask = tokenizedInput.attention_mask.to(device=model.device)

    logging.info("InputIDs Shape: {}".format(inputIDs.shape))

    with torch.no_grad():
        if deepSpeed:
            if not dsModel:
                raise RuntimeError(f"dsModel not passed to infer()!")
            genTokens = dsModel.module.generate(
                input_ids=inputIDs,
                attention_mask=attentionMask,
                max_new_tokens=MODEL_MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                **generationConfig,
            )
        else:
            genTokens = model.generate(
                input_ids=inputIDs,
                attention_mask=attentionMask,
                max_new_tokens=MODEL_MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                **generationConfig,
            )
        if modelName == "gptj": #Causal Language Modelling
            outputIDs = genTokens[0, len(inputIDs[0]):]
        else:   #Seq-2-Seq 
            outputIDs = genTokens[0, :]
        genText = tokenizer.decode(outputIDs)
    return genText 
#---------------------------------------------------------------------------
def performInference(unlabelled, testData, testFewShotPrompt, dataset, modelName, model, tokenizer, trackInference=False, generationConfig={}, maxShots=np.inf, direct=False, deepSpeed=False, dsModel=None):
    if unlabelled:
        accuracyScore = None
    else:
        accuracyScore = 0
    predictions = []
    for testInstance in tqdm(testData, desc="Test Instance"):
        testPrompt = generateTestPrompt(testInstance, dataset, modelName, maxShots, direct, False)        

        # finalPrompt = testFewShotPrompt + testPrompt + "</s>"
        finalPrompt = testFewShotPrompt + testPrompt

        genText = infer(model, model, tokenizer, finalPrompt, generationConfig, deepSpeed, dsModel)

        if trackInference:
            logging.info("Input:\n{}".format(finalPrompt))
            logging.info("Output:\n{}".format(genText))
            logging.info("-"*20)

        extractedAnswer = extractAnswer(genText, dataset, direct)
        if extractedAnswer == None:
            continue
        prediction = extractedAnswer["answer"]
        predictionString = extractedAnswer["answerString"]
        rationale = extractedAnswer["rationale"]
        
        if unlabelled:
            testInstance.update({
                "output": genText,
                "answer": prediction,
                "answerString": predictionString,
                "explanation": rationale
            })
        else: 
            corrAnswer = testInstance["answerKey"]
            if prediction.lower() == corrAnswer.lower():
                accuracyScore += 1
            testInstance.update({
                "modelOutput": genText,
                "modelAnswer": prediction,
                "modelAnswerString": predictionString,
                "modelExplanation": rationale
            })
            
        predictions.append(testInstance)

    if not unlabelled:
        accuracyScore /= len(testData)

    return predictions, accuracyScore
#---------------------------------------------------------------------------
def checkIfExists(path, isDir=False, createIfNotExists=False): 
    if isDir and not path.endswith("/"):
        raise ValueError("Directory path should end with '/'")
    pathExists = exists(path)
    if not pathExists:
        if createIfNotExists:
            os.makedirs(path) 
        else:
            raise ValueError(f"{path} is an invalid path!")
    if not isDir:
        filePath = Path(path)
        if not filePath.is_file():
            raise ValueError(f"{path} is not a file!")
#---------------------------------------------------------------------------
def readFile(filePath):
    if filePath.endswith(".json"):
        with open(filePath, "r") as f:
            data = json.loads(f)
        return data
    elif filePath.endswith(".jsonl"):
        data = []
        with open(filePath, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    else: 
        raise ValueError("File {} has unsupported file extension!".format(filePath))
#---------------------------------------------------------------------------
def main():
    args = parser.parse_args()

    wandb.init(
        project="ZARA",
        config = args,
        allow_val_change=True
    )

    config = wandb.config

    #<Process configuration>
    if config.log:
        if config.log!="stdout" and config.log.endswith(".txt"):
            logging.basicConfig(filename=config.log, filemode='w', level=logging.INFO)
        elif config.log=="stdout":
            logging.basicConfig(filemode='w', level=logging.INFO)
        elif config.log=="none":
            logging.basicConfig(filemode='w', level=logging.ERROR)
        else: 
            raise ValueError("Invalid log file {}!".format(config.log))
    else: 
        logging.basicConfig(filemode='w', level=logging.ERROR)

    if config.maxSteps <= 0:
        logging.warning("maxSteps cannot be non-positive!")
        config.maxSteps = -1

    if config.numEpochs < 1:
        logging.warning("numEpochs cannot be non-positive!")
        config.numEpochs = 1

    if config.episodes <= 1:
        raise ValueError("At least two episodes should be provided to be able to sample unlabelled data instances!")

    if config.endEpisode == -1: 
        config.update({
                    "endEpisode": config.startEpisode + config.episodes
                }, 
                allow_val_change=True
            )
    elif config.endEpisode <= config.startEpisode:
        raise ValueError("endEpisode has to be greater than startEpisode!")
    
    checkIfExists(config.dataPrefix, True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    #</Process configuration>

    #<Read Data>
    trainDataByEpisode = {}
    testDataByEpisode = {}
    for episode in range(config.episodes):
        trainFile = config.dataPrefix+"episode_"+str(episode)+"/"+config.trainName
        testFile = config.dataPrefix+"episode_"+str(episode)+"/"+config.testName
        checkIfExists(trainFile)
        checkIfExists(testFile)
        trainData = pd.DataFrame(readFile(trainFile))
        testData = pd.DataFrame(readFile(testFile))
        trainData["episode"] = episode
        testData["episode"] = episode
        trainDataByEpisode[episode] = trainData
        testDataByEpisode[episode] = testData

    trainDataByEpisodeDF = pd.concat(trainDataByEpisode.values())
    testDataByEpisodeDF = pd.concat(testDataByEpisode.values())
    #</Read Data>

    logging.info("No. of Episodes: {}".format(config.episodes))
    logging.info("Start Episode: {}".format(config.startEpisode))
    logging.info("End Episode: {}".format(config.endEpisode))

    for episode in tqdm(range(config.startEpisode, config.endEpisode), desc="Episode"):
        logging.info("Episode {}/{}".format(episode, config.episodes))

        model, tokenizer, modelID, _ = loadModel(
            model=config.model, 
            size=config.size, 
            device=device,
            train=True, 
            peft=config.peft
        )

        trainData = trainDataByEpisode[episode].to_dict("records")
        testData = testDataByEpisode[episode].to_dict("records")

        testFewShotPrompt=""
        if config.testPrompt!="None":
            if config.testPrompt.endswith(".txt"):
                with open(config.testPrompt, "r") as f:
                    testFewShotPrompt = f.read()
            elif config.testPrompt == "sample":
                testFewShotPrompt = generateTrainPrompt(np.random.choice(trainData, config.maxShots), config.dataset, config.model, config.maxShots, config.direct, False)
            else: 
                raise ValueError("Invalid test prompt: {}".format(config.testPrompt))

        #+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
        #Train M_0 on trainData
        logging.info("+="*30)
        logging.info("Training on train data...")

        trainDataLoader, dataCollator = getTrainDataLoader(
            trainData=trainData, 
            dataset=config.dataset, 
            model=config.model, 
            tokenizer=tokenizer, 
            batchSize=config.batchSize, 
            direct=config.direct, 
            useTrainPrompt=config.trainPrompt
        )

        numTrainingSteps = config.numEpochs * len(trainDataLoader)

        optimizer, lrScheduler, modelEngine, modelOptimizer = setupTrainer(
            model=model, 
            learningRate=config.learningRate, 
            useLrScheduler=config.lrScheduler,
            numTrainingSteps=numTrainingSteps,
            deepSpeed=config.deepSpeedTrain,
            dataCollator=dataCollator,
            deepSpeedConfig=config.deepspeed,
        )

        if config.maxSteps == -1:
            config.update({
                    "maxSteps": numTrainingSteps
                }, 
                allow_val_change=True
            )
        elif config.maxSteps > 0:
            config.update({
                    "numEpochs": math.ceil(config.maxSteps/len(trainDataLoader))
                },
                allow_val_change=True
            )
        else: 
            raise ValueError(f"Maximum no. of steps (maxSteps) has to be positive!")

        saveModelPath = f"{config.savePath}{config.model}_{config.size}_episode_{episode}"

        #Train the model
        performTraining(
            model=model, 
            saveModelPath=saveModelPath, 
            device=device,
            optimizer=optimizer, 
            trainDataLoader=trainDataLoader, 
            numEpochs=config.numEpochs, 
            maxSteps=config.maxSteps, 
            useLrScheduler=config.lrScheduler, 
            lrScheduler=lrScheduler, 
            trackTraining=TRACK_TRAINING, 
            tokenizer=tokenizer,
            deepSpeedTrain=config.deepSpeedTrain, 
            modelEngine=modelEngine
        )
        logging.info("+="*30)
        #+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

        #+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
        #Sample |D_test| "unlabelled" instances from D_test of other episodes for augmentation 
        logging.info("+="*30)
        logging.info("Inference on unlabelled dataset...")

        testDataToSampleFrom = testDataByEpisodeDF[testDataByEpisodeDF["episode"]!=episode].to_dict("records")
        unlabelledTestData = np.random.choice(testDataToSampleFrom, len(testData))

        #Perform inference on this "unlabelled" dataset to obtain rationale-answer pairs
        modelID = f"{config.savePath}{config.model}_{config.size}_episode_{episode}"
        model, tokenizer, _, dsModel = loadModel(
            model=config.model, 
            size=config.size, 
            device=device,
            modelID=modelID, 
            train=False,
            deepSpeed=config.deepSpeedTest, 
            localRank=args.local_rank,
        )
        generationConfig = {
            # "min_length": MINLENGTH,
            # "do_sample":MODEL_DO_SAMPLE,
            # "temperature":MODEL_TEMPERATURE,
            # "top_p":MODEL_TOP_P,
            # "top_k":MODEL_TOP_K,
            # "repetition_penalty":MODEL_REPETITION_PENALTY,
        } 

        unlabelledPredictions, _ = performInference(
            unlabelled=True,
            testData=unlabelledTestData,
            testFewShotPrompt=testFewShotPrompt,
            dataset=config.dataset,
            modelName=config.model,
            model=model,
            tokenizer=tokenizer,
            trackInference=TRACK_INFERENCE,
            generationConfig=generationConfig,
            maxShots=config.maxShots,
            direct=config.direct,
            deepSpeed=config.deepSpeedTest,
            dsModel=dsModel
        )
        logging.info("+="*30)
        #+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

        #+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
        #Convert to NLI formulation
        logging.info("+="*30)
        logging.info("Finding instances to augment train data with...")

        unlabelledNLIprompts = mapToNLI(unlabelledPredictions, config.dataset, config.model, np.inf, isTest=True)
        
        #Perform inference using ensemble 
        instancesToAugment = []
        classifiers = [pipeline('zero-shot-classification', model=model) for model in config.NLImodels]
        with torch.no_grad():
            for unlabelledInstance, unlabelledNLIprompt in tqdm(zip(unlabelledPredictions, unlabelledNLIprompts), desc="NLI on unlabelled dataset"):
                avgScore = 0
                for ind, modelName in enumerate(config.NLImodels):
                    classifier = classifiers[ind]
                    output = classifier(unlabelledNLIprompt, list(NLIlabels.keys()))
                    predictedNLIclass = output["labels"][np.argmax(output["scores"])]
                    if TRACK_NLI:
                        logging.info(f"Prompt to {modelName}:\n\t{unlabelledNLIprompt}")
                        logging.info("{} ({})".format(predictedNLIclass, np.max(output["scores"])))
                    avgScore += output["scores"][output["labels"].index(TargetNLIlabels[config.dataset])]
                avgScore /= len(config.NLImodels)
                if avgScore >= NLIlabels[TargetNLIlabels[config.dataset]][config.dataset]["threshold"]:
                    instancesToAugment.append(unlabelledInstance)
        logging.info("No. of instances to be augmented to train set: {}".format(len(instancesToAugment)))
        
        #Augment highly confident rationale-answer pairs 
        if len(instancesToAugment) == 0:
            logging.info("No instances to augment trainset with! Consider using a lower plausibility threshold.")
            continue
        logging.info("+="*30)
        #+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

        #+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
        #Train M_0 on Augmented trainData
        logging.info("+="*30)
        logging.info("Training on augmented train data...")
        model, tokenizer, modelID, _ = loadModel(
            model=config.model, 
            size=config.size, 
            device=device,
            train=True, 
            peft=config.peft
        )

        trainData = trainDataByEpisode[episode].to_dict("records")
        trainData.extend(instancesToAugment)
        testData = testDataByEpisode[episode].to_dict("records")

        trainDataLoader, dataCollator = getTrainDataLoader(
            trainData=trainData, 
            dataset=config.dataset, 
            model=config.model, 
            tokenizer=tokenizer, 
            batchSize=config.batchSize, 
            direct=config.direct, 
            useTrainPrompt=config.trainPrompt
        )

        numTrainingSteps = numEpochs * len(trainDataLoader)

        optimizer, lrScheduler, modelEngine, modelOptimizer = setupTrainer(
            model=model, 
            learningRate=config.learningRate, 
            useLrScheduler=config.lrScheduler,
            numTrainingSteps=numTrainingSteps,
            deepSpeed=config.deepSpeedTrain,
            dataCollator=dataCollator,
            deepSpeedConfig=config.deepspeed,
        )

        if config.maxSteps == -1:
            config.update({
                    "maxSteps": numTrainingSteps
                }, 
                allow_val_change=True
            )
        elif config.maxSteps > 0:
            config.update({
                    "numEpochs": math.ceil(config.maxSteps/len(trainDataLoader))
                },
                allow_val_change=True
            )
        else: 
            raise ValueError(f"Maximum no. of steps (maxSteps) has to be positive!")

        saveModelPath = f"{config.savePath}{config.model}_{config.size}_episode_{episode}_augmented"

        #Re-train model on augmented train instance
        performTraining(
            model=model, 
            saveModelPath=saveModelPath, 
            device=device,
            optimizer=optimizer, 
            trainDataLoader=trainDataLoader, 
            numEpochs=config.numEpochs, 
            maxSteps=config.maxSteps, 
            useLrScheduler=config.lrScheduler, 
            lrScheduler=lrScheduler, 
            trackTraining=TRACK_TRAINING, 
            tokenizer=tokenizer,
            deepSpeedTrain=config.deepSpeedTrain, 
            modelEngine=modelEngine
        ) 
        logging.info("+="*30)
        #+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+

        #+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
        #Perform inference on test set and report results
        logging.info("+="*30)
        logging.info("Performing inference on test set...") 
        modelID = f"{config.savePath}{config.model}_{config.size}_episode_{episode}_augmented"
        model, tokenizer, _, dsModel = loadModel(
            model=config.model, 
            size=config.size, 
            device=device,
            modelID=modelID, 
            train=False,
            deepSpeed=config.deepSpeedTest, 
            localRank=args.local_rank,
        )
        
        generationConfig = {
            # "min_length": MINLENGTH,
            # "do_sample":MODEL_DO_SAMPLE,
            # "temperature":MODEL_TEMPERATURE,
            # "top_p":MODEL_TOP_P,
            # "top_k":MODEL_TOP_K,
            # "repetition_penalty":MODEL_REPETITION_PENALTY,
        }

        _, accuracyScore = performInference(
            unlabelled=False,
            testData=testData,
            testFewShotPrompt=testFewShotPrompt,
            dataset=config.dataset,
            modelName=config.model,
            model=model,
            tokenizer=tokenizer,
            trackInference=TRACK_INFERENCE,
            generationConfig=generationConfig,
            maxShots=config.maxShots,
            direct=config.direct,
            deepSpeed=config.deepSpeedTest,
            dsModel=dsModel
        )

        logging.info(f"Accuracy on test data: {accuracyScore}")
        wandb.log({
            "test_accuracy": accuracyScore
        })
        logging.info("+="*30)
        #+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
    
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wandb.finish()
        
#---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
