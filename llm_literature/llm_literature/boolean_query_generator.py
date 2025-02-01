from typing import Optional
from pydantic import BaseModel
from openai import OpenAI


class BooleanQuery(BaseModel):
    keyword: str
    booleanquery: str

class BooleanQueryGenerator:
    def __init__(self, review_title: str, model: str = "qwen2.5:7b", base_query: Optional[str] = None):
        """
        Initialize the BooleanQueryGenerator instance.

        Parameters:
            review_title (str): The title used for generating the query.
            model (str): The LLM model to use.
            base_query (Optional[str]): A predefined base Boolean query.
        """
        self.review_title = review_title
        self.model = model
        self.base_query = base_query or (
            '("large language models" OR "large language model" OR "LLM" OR "LLMs" OR "ChatGPT" OR '
            '"GPT-3" OR "GPT-4" OR "LLaMA" OR "Mistral" OR "Mixtral" OR "BARD" OR "BERT" OR "Claude" '
            'OR "PaLM" OR "Gemini" OR "Copilot") AND '
            '("systematic review*" OR "scoping review*" OR "literature review*" OR "narrative review*" '
            'OR "umbrella review*" OR "rapid review*" OR "integrative review*" OR "evidence synthesis" '
            'OR "meta-analysis")'
        )

    def get_prompt(self) -> str:
        """
        Generate a prompt for interacting with the LLM.

        Returns:
            str: The complete prompt.
        """
        return (
            f"# Requirment\n You are an information specialist who develops Boolean queries for systematic reviews. "
            f"Your specialty is creating highly effective queries to retrieve relevant academic literature. "
            f"You excel at balancing precision and recall in Boolean queries. "
            f"Futhermore, you should summarize a keyword that include each key point in user's query"
            f"# Example\n Given the information need 'The emergence of Large Language Models (LLM) as a tool in literature reviews', "
            f"Example boolean query: '{self.base_query}'\n Example keyword: 'Literature review using Large Language Model' "
            # f"Based on the information need '{self.review_title}',please generate a highly effective Boolean query and its keyword."
        )

    def get_boolean_query(self, client: OpenAI) -> str:
        """
        Get the final Boolean query.

        Returns:
            str: The Boolean query.
        """
        prompt = self.get_prompt()

        response = client.beta.chat.completions.parse(
        model="qwen2.5:7b",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Based on the information need '{self.review_title}',please generate a highly effective Boolean query and its keyword."},
            ],
        response_format=BooleanQuery
        )
        
        return dict(response.choices[0].message.parsed)