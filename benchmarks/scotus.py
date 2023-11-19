from models.rag_model import RagModel
from datasets import load_dataset
import json

TEMPLATE = """Below is a description of a court opinion. Please select the category that best matches the relevant issue area. The possible issue areas are:

- Criminal Procedure
- Civil Rights
- First Amendment
- Due Process
- Privacy
- Attorneys
- Unions
- Economic Activity
- Judicial Power
- Federalism
- Interstate Relations
- Federal Taxation
- Miscellaneous
- Private Action

Context of the case:
{question}

Additionally, you might consider paying attention to some of the law books:
{context}

The most relevant issue area for this case is """


NUMBER_TO_LABEL = {
    0: 'Criminal Procedure',
    1: 'Civil Rights',
    2: 'First Amendment',
    3: 'Due Process',
    4: 'Privacy',
    5: 'Attorneys',
    6: 'Unions',
    7: 'Economic Activity',
    8: 'Judicial Power',
    9: 'Federalism',
    10: 'Interstate Relations',
    11: 'Federal Taxation',
    12: 'Miscellaneous',
    13: 'Private Action'
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
    pdf = '../data/basic-laws-book-2016.pdf'
    vector_store_name = "basic-laws"
    rag = RagModel(
        model_name=model_name,
        pdf=pdf,
        store_model_name=store_model_name,
        vector_store_name=vector_store_name,
        template=TEMPLATE
    )

    dataset = load_dataset("lex_glue", "scotus")['test']

    results = {}
    for i in range(len(dataset)):
        print(f"Running {i} of {len(dataset)}")
        text = dataset[i]['text'][:1000]
        number = dataset[i]['label']
        label = NUMBER_TO_LABEL[number]
        rag_label = rag(text)

        rag_number = get_number(rag_label)
        results[i] =  { 'true_label': number, 'rag_label': rag_number }

    # save json
    with open('rag_results_scotus.json', 'w') as f:
        json.dump(results, f)



