import openai
from typing import List
import os

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"


class LLMClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str,
        max_tokens: int = 4096,
        temperature: float = 1.2,
        top_p: float = 0.1,
        n: int = 1,
    ):
        self.model = model
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n

    def response_from_list(self, msgs: List[dict]):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            # n=self.n,
        )

        return response.choices[0].message.content

    def response_from_question(self, question: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": question},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
        )

        return response.choices[0].message.content


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


# compare two documents using n-gram similarity
def generate_ngrams(text, n):
    ngrams = set()
    for i in range(len(text) - n + 1):
        ngram = text[i : i + n]
        ngrams.add(ngram)
    return ngrams


def ngram_similarity(doc1, doc2, n=2):
    ngrams_doc1 = generate_ngrams(doc1, n)
    ngrams_doc2 = generate_ngrams(doc2, n)
    return jaccard_similarity(ngrams_doc1, ngrams_doc2)
