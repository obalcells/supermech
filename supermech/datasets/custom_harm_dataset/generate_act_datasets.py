import json
from random import shuffle

# generates an A/B dataset from a dataset of questions and answers
# the format of the file 'data_path' must match the format of the file 'datasets/nina_rimsky_refusal_data/refusal_data.json'
def gen_A_B_dataset(
    data_path,
    output_path,
    question_key="question",
    answer_matching_behavior_key="decline_answer", # arg example from refusal dataset
    answer_not_matching_behavior_key="respond_answer",
):
    A_B = []
    with open(data_path, "r") as f:
        data = json.load(f)
        for item in data:
            item[question_key] = item[question_key].strip()
            item[answer_matching_behavior_key] = item[answer_matching_behavior_key].strip()
            item[answer_not_matching_behavior_key] = item[answer_not_matching_behavior_key].strip()
            answers = [item[answer_matching_behavior_key], item[answer_not_matching_behavior_key]]
            # we shuffle the answers such that both (A) and (B) are equally likely to be the answer matching the behavior
            if answers[0] == item[answer_matching_behavior_key]:
                answer_matching_behavior = "(A)"
                answer_not_matching_behavior = "(B)"
            else:
                answer_matching_behavior = "(B)"
                answer_not_matching_behavior = "(A)"
            full_question = (
                f"{item[question_key]}\n\nChoices:\n(A) {answers[0]}\n(B) {answers[1]}"
            )
            A_B.append(
                {
                    "question": full_question,
                    "answer_matching_behavior": answer_matching_behavior,
                    "answer_not_matching_behavior": answer_not_matching_behavior,
                }
            )

    with open(output_path, "w") as f:
        json.dump(A_B, f)