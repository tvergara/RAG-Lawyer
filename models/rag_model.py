from models.base_model import BaseLLM
from models.base_vector_store import get_vector_store
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import torch
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


class RagModel:
    def __init__(
            self,
            model_name: str,
            store_model_name: str,
            pdfs: list[str],
            vector_store_name: str,
            template: str,
            device: str
    ):
        self.llm = BaseLLM(model_name=model_name, device=device)
        self.vector_store = get_vector_store(
            model_name=store_model_name,
            name=vector_store_name,
            pdfs=pdfs
        )
        template = template
        chain_prompt = PromptTemplate.from_template(template)
        self.chain = load_qa_chain(self.llm, chain_type="stuff", prompt=chain_prompt)

    def __call__(self, prompt: str, k=1) -> str:
        docs = self.vector_store.similarity_search(prompt, k=k)
        return self.chain(
            {"input_documents": docs, "question": prompt},
            return_only_outputs=True
        )['output_text']

if __name__ == "__main__":
    model_name = "llmware/bling-sheared-llama-1.3b-0.1"
    store_model_name = "thenlper/gte-base"
    pdfs=['./data/basic-laws-book-2016.pdf', './data/european-human-rights.pdf']
    vector_store_name = "basic-laws"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    template = """Use the following pieces of context to answer the question at the end.
{context}
Question: {question}
Helpful Answer: """
    rag = RagModel(
        model_name=model_name,
        pdfs=pdfs,
        store_model_name=store_model_name,
        vector_store_name=vector_store_name,
        template=template,
        device=device
    )
    print(rag("What year was the Atomic Energy Act created?"))
