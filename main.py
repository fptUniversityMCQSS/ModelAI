"""
Google Colab Example: https://colab.research.google.com/github/UKPLab/sentence-transformers/blob/master/examples/applications/retrieve_rerank/retrieve_rerank_simple_wikipedia.ipynb
"""
import json
import os
import re
from model.answer import Answer
from model.comparer import Comparer
from model.document import Document, list_documents, from_pdf_to_document
from entity.question import Question, Option, AnsweringResponse
from model.retriever import Retriever
from flask import stream_with_context
from flask_restful import Resource, Api
from flask import Flask, flash, request, redirect, url_for, jsonify

retriever = Retriever()
comparer = Comparer()
documents = list_documents()


def solve_questions(questions):
    for question in questions:
        query = re.sub(r"\.{5,}", " what ", question.content)
        query = re.sub(r"\s+", " ", query.strip())
        best_answer = None
        for document in documents:
            try:
                retriever.load(document)
            except:
                continue

            answer = retriever.search(query)
            for option in question.options:
                score = comparer.compare(option.content, answer.content)
                if best_answer is None:
                    best_answer = Answer(score, option.key)
                elif best_answer.score < score:
                    best_answer.score = score
                    best_answer.content = option.key

            question.answer = best_answer.content
            print(f"-> Answer: {question.answer}\n")


# db_connect = create_engine('sqlite:///chinook.db')
app = Flask(__name__)
api = Api(app)


@app.route('/knowledge', methods=['POST'])
def streamed_response():
    def generate():
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        yield 'uploaded success, encoding...'
        print(f"creating...")
        document = from_pdf_to_document(file.stream, file.filename)
        print(f"created {document.path_txt} success, encoding...")
        retriever.encode(document)
        print(f"encoded {document.path_pt} success")
        yield 'encoded success'

    return app.response_class(stream_with_context(generate()))


@app.route('/qa', methods=['POST'])
def qa_res():
    json_questions = json.loads(json.dumps(request.json))
    questions = []
    answers_response = []
    for json_question in json_questions:
        options = []
        for json_option in json_question['options']:
            option = Option(json_option['key'], json_option['content'])
            options.append(option)

        question = Question(
            json_question['qn'],
            json_question['content'],
            options)
        questions.append(question)

    solve_questions(questions)

    for question in questions:
        answers_response.append(AnsweringResponse(question.qn, question.answer))

    json_string = json.dumps([ob.__dict__ for ob in answers_response])
    return json_string


if __name__ == '__main__':
    app.run(port='5002')
