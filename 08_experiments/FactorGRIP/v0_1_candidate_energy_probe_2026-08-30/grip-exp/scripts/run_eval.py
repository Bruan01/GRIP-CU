from typing import Optional

from arguments import (
    parse_args,
    BaseArguments,
    EvalArguments,
    InferenceArguments,
    ModelArguments,
    TaskArguments,
)
from constants import SYSTEM_PROMPT
from evaluation.utils import normalize_answer
from models import get_inf_model
from utils import load_list_json, set_random_seed, save_list_json


DATA_FILE = "outputs/data/scene_graph/processed_test.json"


def do_llm_evaluation(
        dataset: str,
        input_file: str,
        output_file: str,
        exp_args: dict,
        llm_as_judge_model: Optional[str] = None,
):
    results = load_list_json(input_file)
    data_file = f"outputs/data/{dataset}/processed_test.json"
    data = load_list_json(data_file)
    data_dict = {}
    for d in data:
        data_dict[d["id"]] = d
    targets = []
    preds = []
    preds_no_sample = []
    questions = []
    for result in results:
        targets.append(result["target"])
        preds.append(result["response"])
        questions.append(result["question"])
        if "response_no_sample" in result:
            preds_no_sample.append(result["response_no_sample"])

    eval_model = get_inf_model(llm_as_judge_model, **exp_args)
    user_prompt = """
    You will be given one question, one predicted answer from an llm candidate model, and the ground truth answers. 
    Be as best as you can to evaluate whether the answer generated from the model match one of the ground truth. If 
    be included, response yes, otherwise no. A match is not required to be exactly the same, 
    semantically included can be regarded as a match.
    Please DON'T output quotes when outputting your evaluation. 
    Here is some examples:
    --Example 1--
    Question: What is in the box?
    Candidate answer: a pizza slice.
    Ground truth: [pizza, pizza slice].
    Evaluation: yes

    --Example 2--
    Question: On which side of the image are the chairs?
    Candidate answer: The chairs are on the left side of the image.
    Ground truth: [right]
    Evaluation: no


    --Example 3--
    Question: Who is wearing the hat?
    Candidate answer: the guy at coordinate (410, 163)
    Ground truth: [guy]
    Evaluation: yes

    --Real task--
    Question: {question}
    Candidate answer: {pred}
    Ground truth: {target}
    Evaluation: 
    """
    if preds_no_sample:
        user_content = [user_prompt.format(question=q, pred=p, target=t) for q, p, t in zip(questions, preds_no_sample, targets)]
        eval_results = eval_model.inference(user_contents=user_content, system_prompt=SYSTEM_PROMPT)
        scores = [normalize_answer(r["response"]) == "yes" for r in eval_results]
        print(f"llm no sampling accuracy: {sum(scores) / len(scores)}")

    user_content = [user_prompt.format(question=q, pred=p, target=t) for q, p, t in zip(questions, preds, targets)]
    eval_results = eval_model.inference(user_contents=user_content, system_prompt=SYSTEM_PROMPT)
    scores = [normalize_answer(r["response"]) == "yes" for r in eval_results]
    print(f"llm accuracy: {sum(scores) / len(scores)}")


    wrong_result_dict = {}
    for score, result in zip(scores, results):
        if score == 0.0:
            id = result["id"]
            if id not in wrong_result_dict:
                wrong_result_dict[id] = {
                    "id": id,
                    "graph": data_dict[id]["graph"],
                    "question": [result["question"]],
                    "target": [result["target"]],
                    "response": [result["response"]],
                }
            else:
                wrong_result_dict[id]["question"].append(result["question"])
                wrong_result_dict[id]["target"].append(result["target"])
                wrong_result_dict[id]["response"].append(result["response"])
    wrong_result_list = list(wrong_result_dict.values())
    save_list_json(output_file, wrong_result_list)


if __name__ == "__main__":
    _ = set_random_seed(123)
    base_args, exp_args = parse_args(
        (
            BaseArguments,
            (EvalArguments, ModelArguments, InferenceArguments, TaskArguments)),
        no_dict=(BaseArguments,)
    )

    exp_args.pop("model_name")

    do_llm_evaluation(
        dataset=exp_args["dataset_name"],
        input_file=base_args.input_file,
        output_file=base_args.output_file,
        llm_as_judge_model=exp_args["llm_as_judge_model"],
        exp_args=exp_args,
    )