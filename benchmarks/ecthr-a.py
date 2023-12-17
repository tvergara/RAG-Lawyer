from models.rag_model import RagModel
from datasets import load_dataset
import json
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


TEMPLATE = """Below is a list of facts in a court case. After that, there is a human right from the human rights court. You will recognize if it was violated or not.

Additionally, you might consider paying attention to some extracts of the human rights court:
{context}

Context of the case:
{question}"""


QUESTION_TEMPLATE = """{question}

The right called "{right}" was violated? (yes/no)
Answer:"""


NUMBER_TO_LABEL = {
    0: 'Right to life',
    1: 'Prohibition of torture',
    2: 'Right to liberty and security',
    3: 'Right to a fair trial',
    4: 'Right to respect for private and family life',
    5: 'Freedom of thought, conscience and religion',
    6: 'Freedom of expression',
    7: 'Freedom of assembly and association',
    8: 'Prohibition of discrimination',
    9: 'Protection of property',
}
LABEL_TO_NUMBER = {v: k for k, v in NUMBER_TO_LABEL.items()}

def get_number(response: str) -> int:
    response = response.strip()

    label = None
    for key in LABEL_TO_NUMBER.keys():
        if key.lower() in response.lower():
            label = key
            break

    if label not in LABEL_TO_NUMBER.keys():
        print(f"Could not find label in response: {response}")
        return -1

    return LABEL_TO_NUMBER[label]

if __name__ == "__main__":
    model_name = "llmware/bling-sheared-llama-1.3b-0.1"
    store_model_name = "thenlper/gte-base"
    pdfs=['./data/basic-laws-book-2016.pdf', './data/european-human-rights.pdf']
    vector_store_name = "basic-laws"
    rag = RagModel(
        model_name=model_name,
        pdfs=pdfs,
        store_model_name=store_model_name,
        vector_store_name=vector_store_name,
        template=TEMPLATE
    )

    dataset = load_dataset("lex_glue", "ecthr_a")['test']

    print(dataset)

    results = {}
    for i in range(20):
        text = '\n'.join(dataset[i]['text'])
        numbers = []
        for number, right in NUMBER_TO_LABEL.items():
            full_text = QUESTION_TEMPLATE.format(question=text, right=right)
            rag_label = rag(text)
            print(rag_label)
            break

        break




