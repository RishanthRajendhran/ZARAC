import random
import os
import json
import argparse

def to_list(num):
    list_str = list(str(num))
    return list_str

def to_str(num):
    return " ".join(to_list(num))

n_samples = 50000
# n_samples = 2000
context_examples = 10
examples_per_prompt = 1
# examples_per_prompt = 10

val_start = 6
max_digits = 10
include_scrachpad = True
# include_scrachpad = False
fixed_examples = True
# fixed_examples = False
randomized_digits = False
# randomized_digits = True

def gen_examples(n_examples, digits, randomized_digits=False, min_digit=1, max_digit=8):
    complete_str = []
    data = []
    for _ in range(n_examples):
        if randomized_digits:
            digits = random.randrange(min_digit + 1, max_digit)
        first_number = random.randrange(int(10 ** digits), int(10 ** (digits + 1)))
        second_number = random.randrange(int(10 ** digits), int(10 ** (digits + 1)))
        input_sum = f'{to_str(first_number)} + {to_str(second_number)}'
        resultant_str = f'Input:\n{input_sum}\nTarget:\n'
        if include_scrachpad:
            scratch_pad = f'<scratch>\n{input_sum} , C: 0\n'
            carry = 0
            running_sum = ''
            initial = True
            for first_digit, second_digit in reversed(list(zip(
                to_list(first_number), to_list(second_number)
            ))):
                dig_sum = int(first_digit) + int(second_digit) + carry
                if not initial:
                    scratch_pad += f'{first_digit} + {second_digit} , {running_sum}C: {carry}\n'
                carry = int(dig_sum >= 10)
                running_sum = f'{dig_sum % 10} {running_sum}'
                initial = False
            scratch_pad += f', {running_sum}C: {carry}\n'
            scratch_pad += f'{carry} {running_sum}'.strip() + '\n'
            scratch_pad += '</scratch>\n'
            resultant_str += scratch_pad
        resultant_str += to_str(first_number + second_number)
        resultant_str += '\n\n'
        complete_str.append(resultant_str)
        dataInstance = {
            "question": input_sum,
            "answerKey": to_str(first_number + second_number),
        }
        if include_scrachpad:
            dataInstance.update({
                "scratch": scratch_pad,
            })
        data.append(dataInstance)
    return complete_str, data


if fixed_examples and randomized_digits:
    few_shot_str, _ = gen_examples(examples_per_prompt - 1, None, randomized_digits, 1, val_start - 1)
for digits in range(max_digits):
    folder_name = 'val' if digits + 1 >= val_start else 'train'
    if include_scrachpad:
        folder_name += '_scratch'
        if fixed_examples and not randomized_digits:
            few_shot_str, _ = gen_examples(context_examples, digits, randomized_digits)
    else:
        folder_name += '_direct'
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    complete_str = ''
    allData = []
    for _ in range(n_samples):
        n_gen_examples = 1 if fixed_examples else examples_per_prompt
        if fixed_examples:
            complete_str += "".join(random.sample(few_shot_str, examples_per_prompt - 1))
        prompts, data = gen_examples(n_gen_examples, digits)
        allData.extend(data)
        complete_str += "".join(prompts)
        complete_str += '<|endoftext|>'

    with open(f'{folder_name}/{digits + 1}.txt', 'w') as f:
        f.write(complete_str)

    with open(f'{folder_name}/{digits + 1}.json', 'w') as f:
        json.dump(allData, f)

    if fixed_examples:
        with open(f'{folder_name}/fewShot_{digits + 1}.txt', 'w') as f:
            f.write("".join(few_shot_str))