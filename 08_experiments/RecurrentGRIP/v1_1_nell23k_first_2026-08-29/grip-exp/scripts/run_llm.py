import os

from tqdm import tqdm
from transformers import TrainingArguments

from arguments import (
    BaseArguments,
    TaskArguments,
    ModelArguments,
    CustomTrainingArguments,
    InferenceArguments,
    parse_args
)
from constants import SYSTEM_PROMPT
from grip.tasks import gen_eval_task
from models import get_inf_model
from utils import extract_tag_content, save_entry_to_list_json, load_list_json, set_random_seed, Timer


def extract_all_evidence_and_answers(text, add_evidence=True):
    if add_evidence:
        evidences = extract_tag_content(text, "evidence")
        evidences = "; ".join(evidences)
    else:
        evidences = ""

    answers = extract_tag_content(text, "answer")
    answers = "; ".join(answers)
    return evidences, answers


def run(
        rank: int,
        input_data: list[dict],
        exp_args: dict,
        output_file: str,
        num_resumed: int = 0,
):
    # get model
    model = get_inf_model(**exp_args)
    user_contents = []
    question_list = []
    answer_list = []
    id_list = []
    token_counts = []
    # start training
    for i, input_d in enumerate(tqdm(input_data, desc=f"Sample: ")):
        if i < num_resumed:
            continue
        id = input_d["id"]
        eval_dataset = gen_eval_task(
            eval_mode="standard",
            input_data=input_d,
            **exp_args,
        )
        for user_content, q, a in eval_dataset:
            if exp_args["report_input_token_count"]:
                tokens = model.tokenizer.encode(user_content)
                token_counts.append(len(tokens))
            user_contents.append(user_content)
            question_list.append(q)
            answer_list.append(a)
            id_list.append(id)
    if exp_args["report_input_token_count"]:
        print(f"averge input token count: {sum(token_counts) / len(token_counts)}")
    timer = Timer()
    timer.start()
    results = model.inference(user_contents=user_contents, system_prompt=SYSTEM_PROMPT)
    timer.end()
    print(f"Total inference time:{timer.return_time()}")

    for result, Q, A, id in zip(results, question_list, answer_list, id_list):
        response = result["response"]

        evidence, answer = extract_all_evidence_and_answers(response, False)
        output_result = {
            "id": id,
            "question": Q,
            "response": answer.strip(),
            "evidence": evidence.strip(),
            "target": [A]}
        # save
        save_entry_to_list_json(output_file, output_result)


def main():
    _ = set_random_seed()
    base_args, exp_args = parse_args(
        (
            BaseArguments,
            (TaskArguments, ModelArguments, CustomTrainingArguments, InferenceArguments)),
        no_dict=(TrainingArguments,)
    )
    # set up experiment
    rank = base_args.pop("rank")
    input_file = base_args.pop("input_file")
    output_file = base_args.pop("output_file")
    overwrite = base_args.pop("overwrite")

    num_resumed = 0
    if os.path.exists(output_file):
        if overwrite:
            os.remove(output_file)
        else:
            with open(output_file, 'r') as f:
                num_resumed = len(f.readlines())
    input_data = load_list_json(input_file)

    run(rank, input_data, exp_args, output_file, num_resumed)


if __name__ == "__main__":
    main()
