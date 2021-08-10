# https://huggingface.co/transformers/task_summary.html#extractive-question-answering

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from model.result import Result
import torch

# question answering model
model = "bert-large-uncased-whole-word-masking-finetuned-squad"


class QuestionAnswering:
    def __init__(self):
        # self.model = AutoModelForQuestionAnswering.from_pretrained(model)
        # self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.qa = pipeline("question-answering")

    def answer(self, question, context):
        result = self.qa(question=question, context=context)
        # print(f"QA: '{result['answer']}', score: {round(result['score'], 3)}")
        return Result(result['score'], result['answer'])

        # inputs = self.tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
        # input_ids = inputs["input_ids"].tolist()[0]
        # outputs = self.model(**inputs)
        # answer_start_scores = outputs.start_logits
        # answer_end_scores = outputs.end_logits
        # answer_start = torch.argmax(
        #     answer_start_scores
        # )  # Get the most likely beginning of answer with the argmax of the score
        # answer_end = torch.argmax(
        #     answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
        # answer = self.tokenizer.convert_tokens_to_string(
        #     self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        # print(f"QA model: Answer: {answer}")
        # return Result()