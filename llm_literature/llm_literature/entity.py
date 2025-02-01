import pandas as pd
import numpy as np
import voyageai
import ast
import os
from .author_article_manager import AuthorArticleManager  


def get_embedding(text, embedding_client: voyageai.Client):
    """
    Gets the embedding for a given text using the specified embedding client.

    Args:
        text (str): The text to embed.
        embedding_client (voyageai.Client): The Voyage AI client to use for embedding.

    Returns:
        list: The embedding vector for the text.

    Raises:
        ValueError: If the embedding client is not initialized properly.
    """
    if not embedding_client:
        raise ValueError("Error: Embedding client is not initialized.")

    try:
        vo = embedding_client
        return vo.embed(texts=text, model="voyage-3-lite").embeddings
    except Exception as e:
        raise RuntimeError(f"Error during embedding generation: {e}")


class Entity:
    """
    Represents an entity with a name, description, sources, and optional name embedding.
    """

    embedding_client = None

    @classmethod
    def set_embedding_client(cls, client):
        """
        Sets the embedding client for the Entity class.

        Args:
            client (voyageai.Client): The Voyage AI client to set.
        """
        cls.embedding_client = client

    def __init__(self, name, description, sources, name_embedding=None):
        """
        Initializes an Entity object.

        Args:
            name (str): The name of the entity.
            description (str): The description of the entity.
            sources (list): A list of DOIs (Digital Object Identifiers) associated with the entity.
            name_embedding (np.ndarray, optional): The embedding vector for the entity's name. Defaults to None.
        """
        self.name = name
        self.description = description
        self.sources = sources
        self.name_embedding = name_embedding

    def __repr__(self):
        """
        Returns a string representation of the Entity object.

        Returns:
            str: A string representation of the Entity.
        """
        return f"Entity(name='{self.name}', description='{self.description}', source='{self.sources}')"
    
    
    def link_to_articles(self, manager: "AuthorArticleManager"):
        """
        Links the current Entity to the Articles in AuthorArticleManager based on DOI.

        Args:
            manager (AuthorArticleManager): The AuthorArticleManager instance containing articles.

        Raises:
            TypeError: If the manager is not an instance of AuthorArticleManager.
        """

        if not isinstance(manager, AuthorArticleManager):
            raise TypeError("Error: 'manager' must be an instance of AuthorArticleManager.")
            
        for article in manager.articles.values():
            for doi in self.sources:
                if article.doi == doi:
                    # Add this entity to the article's entities list
                    article.entities.append(self)


    @staticmethod
    def read_name_description_from_parquet(file_path="./output"):
        """
        Reads entity data from parquet files and creates a list of Entity objects.

        Args:
            file_path (str): The path to the directory containing the parquet files. Defaults to "./output".

        Returns:
            list: A list of Entity objects.

        Raises:
            ValueError: If the API key is not set.
            FileNotFoundError: If any of the required parquet files are not found.
            RuntimeError: If there is an error during the process.
        """


        entity_dir = os.path.join(file_path, "create_final_entities.parquet")
        text_dir = os.path.join(file_path, "create_final_text_units.parquet")
        input_dir = os.path.join(file_path, "input.parquet")

        if not os.path.exists(entity_dir):
            raise FileNotFoundError(f"Error: Entity file not found at {entity_dir}")
        if not os.path.exists(text_dir):
            raise FileNotFoundError(f"Error: Text file not found at {text_dir}")
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Error: Input file not found at {input_dir}")

        try:
            entity_info = pd.read_parquet(entity_dir)
            text_info = pd.read_parquet(text_dir)
            input_info = pd.read_parquet(input_dir)
        except Exception as e:
            raise RuntimeError(f"Error reading parquet files: {e}")

        entities = []

        text_input_map = {}
        for index, row in text_info.iterrows():
            text_input_map[row["id"]] = row["document_ids"][0]

        input_doi_map = {}
        for index, row in input_info.iterrows():
            input_doi_map[row["id"]] = row["source"]

        embedding_file_path = "./cache/text_embedding/embedding_csv.csv"
        os.makedirs(os.path.dirname(embedding_file_path), exist_ok=True)

        if os.path.exists(embedding_file_path):
            cached_embeddings = pd.read_csv(embedding_file_path)
        else:
            cached_embeddings = pd.DataFrame(columns=["name", "embedding"])

        for index, row in entity_info.iterrows():
            try:
                source = []
                for text in row["text_unit_ids"]:
                    source.append(input_doi_map[text_input_map[text]])

                if row["title"] in cached_embeddings["name"].values:
                    embedding = np.array(
                        [
                            ast.literal_eval(
                                cached_embeddings[cached_embeddings["name"] == row["title"]][
                                    "embedding"
                                ].to_list()[0]
                            )
                        ]
                    )

                else:
                    embedding = get_embedding(text=row["title"], embedding_client=voyageai.Client())
                    new_row = {"name": row["title"], "embedding": embedding}
                    
                    # Use pd.concat instead of deprecated .append
                    cached_embeddings = pd.concat([cached_embeddings, pd.DataFrame([new_row])], ignore_index=True)
                    cached_embeddings.to_csv(embedding_file_path, index=False)

                entity = Entity(
                    name=row["title"],
                    description=row["description"],
                    sources=source,
                    name_embedding=np.array(embedding),
                )

                entities.append(entity)
            except Exception as e:
                raise RuntimeError(f"Error processing entity '{row['title']}': {e}")

        return entities
