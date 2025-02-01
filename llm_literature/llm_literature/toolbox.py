import os
import string
import json
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional
from pybibx.base import pbx_probe
import matplotlib.pyplot as plt
from .author_article_manager import AuthorArticleManager
from .data_processor import CSVProcessor
from .graph_builder import GraphBuilder
from .boolean_query_generator import BooleanQueryGenerator
from .entity import Entity, get_embedding


class Toolbox:
    """
    A toolbox for analyzing scientific literature data, providing functionalities like:
    - Finding related CSV files based on a research field.
    - Loading and processing data from CSV files.
    - Building and analyzing author collaboration graphs.
    - Generating boolean queries for literature search.
    - Finding top authors and their collaborations.
    - Finding information about a specific author.
    - Finding entities related to a given name.
    """

    def __init__(self, field: str, database: str = "scopus"):
        """
        Initializes the Toolbox, processing CSV files related to the specified field.

        Args:
            field (str): The name of the field to generate boolean queries for.
            database (str): The name of the database, defaults to 'scopus'.

        Raises:
            TypeError: If `field` is not a string or `database` is not a string.
        """
        if not isinstance(field, str):
            raise TypeError("Error: 'field' must be a string.")
        if not isinstance(database, str):
            raise TypeError("Error: 'database' must be a string.")

        self.field = field
        self.database = database
        self.cleaned_field = self._clean_string(field)
        self.matched_csv = self._find_matched_csv()
        self.data = self._load_data()
        self.manager = AuthorArticleManager(self.data)
        self.graph_builder = GraphBuilder(self.manager)
        self.author_graph = self.graph_builder.build_author_collaboration_graph()

    @staticmethod
    def _clean_string(s: str) -> str:
        """
        Helper function: Replaces spaces with underscores, removes all punctuation (keeping underscores) and converts to lowercase.

        Args:
            s (str): The string to clean.

        Returns:
            str: The cleaned string.

        Raises:
            TypeError: If the input is not a string.
        """
        if not isinstance(s, str):
            raise TypeError("Error: Input must be a string.")

        s = s.replace(" ", "_")
        punctuation = string.punctuation.replace("_", "")
        translator = str.maketrans("", "", punctuation)
        return s.translate(translator).lower()

    def search_boolean_query(self, query):
        """
        Prints the given boolean query. In a real application, this would be replaced with code to search a database.

        Args:
            query (str): The boolean query to search.
        
        Raises:
            NotImplementedError: This function is just a placeholder.
        """
        print(query)
        raise NotImplementedError("Error: `search_boolean_query` is a placeholder function and needs to be implemented "
                                  "with actual database search logic.")

    def _find_matched_csv(self) -> Optional[str]:
        """
        Finds a CSV file that matches the field name. If not found, generates a boolean query and searches.

        Returns:
            Optional[str]: The matched CSV filename or the filename found by boolean query.

        Raises:
            FileNotFoundError: If the 'input' directory does not exist or is not a directory.
            RuntimeError: If no matching CSV file is found and the boolean query search fails.
        """

        csv_files = []
        input_dir = "./input"

        if os.path.exists(input_dir) and os.path.isdir(input_dir):
            csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
        else:
            raise FileNotFoundError(
                f"Error: Directory '{input_dir}' not found or is not a directory."
            )

        for csv_file in csv_files:
            filename_without_ext = os.path.splitext(csv_file)[0]
            cleaned_filename = self._clean_string(filename_without_ext)
            similarity = self._calculate_similarity(cleaned_filename, self.cleaned_field)

            if similarity >= 0.3:
                return csv_file

        # If no matching CSV file is found, generate a boolean query
        boolean_query_generator = BooleanQueryGenerator(review_title=self.field)
        boolean_query = boolean_query_generator.get_boolean_query()
        try:
            matched_csv = self.search_boolean_query(boolean_query["booleanquery"])
        except Exception as e:
            raise RuntimeError(
                f"Error during boolean query search: {e}"
            )
        return matched_csv

    def _calculate_similarity(self, str1, str2):
        """
        Calculates the similarity ratio between two strings.

        Args:
            str1 (str): The first string.
            str2 (str): The second string.

        Returns:
            float: The similarity ratio between the two strings.

        Raises:
            TypeError: If either input is not a string.
        """
        if not isinstance(str1, str) or not isinstance(str2, str):
            raise TypeError("Error: Both inputs must be strings.")
        str1 = str1.lower()
        str2 = str2.lower()
        # Use the shorter string's length for calculating the percentage
        min_length = min(len(str1), len(str2))
        if min_length == 0:
            return 0.0

        matches = 0
        for i in range(min_length):
            if str1[i] == str2[i]:
                matches += 1
        return matches / min_length

    def _load_data(self) -> pd.DataFrame:
        """
        Loads the data using CSVProcessor.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            FileNotFoundError: If no CSV file is found or generated for the field.
            RuntimeError: If there is an error loading the data.
        """

        if not self.matched_csv:
            raise FileNotFoundError(
                f"Error: No CSV file found or generated for field '{self.field}'."
            )

        try:
            csv_processor = CSVProcessor(os.path.join("./input", self.matched_csv))
            return csv_processor.load_data()
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")

    def _save_figure_to_svg(self, filename="output.svg"):
        """
        Captures the current matplotlib figure and saves it as an SVG file.

        Args:
            filename (str): The filename to save the figure to, defaults to "output.svg".

        Raises:
            TypeError: If the filename is not a string.
            RuntimeError: If there is an error saving the figure.
        """
        if not isinstance(filename, str):
            raise TypeError("Error: 'filename' must be a string.")

        fig = plt.gcf()  # Get the current Figure object
        # Check if the Figure object is valid
        if fig.axes:  # Check if the Figure contains Axes objects (i.e., if there is any drawing content)
            try:
                fig.savefig(filename, format="svg")
            except Exception as e:
                raise RuntimeError(f"Error saving figure: {e}")
        else:
            print("Warning: No figure to save, please make sure the drawing function is called.")

    def save_network_collab_output(
        self, network_collab_func, *args, filename="output.svg", **kwargs
    ):
        """
        Calls the network_collab_func function and saves its output graph as an SVG file.

        Args:
            network_collab_func (function): The network_collab function.
            *args: Positional arguments to pass to network_collab_func.
            filename (str): The filename to save the figure to, defaults to "output.svg".
            **kwargs: Keyword arguments to pass to network_collab_func.

        Raises:
            TypeError: If `filename` is not a string or `network_collab_func` is not callable.
            RuntimeError: If there is an error during the execution of `network_collab_func`.
        """

        if not isinstance(filename, str):
            raise TypeError("Error: 'filename' must be a string.")
        if not callable(network_collab_func):
            raise TypeError("Error: 'network_collab_func' must be a callable function.")

        original_show = plt.show  

        def mock_show():
            self._save_figure_to_svg(filename)
            plt.show = original_show
            # plt.show()

        plt.show = mock_show  # Replace plt.show()

        try:
            network_collab_func(*args, **kwargs)  # Call the network_collab function
        except Exception as e:
            raise RuntimeError(f"Error executing network_collab_func: {e}")

    def show_author_collaboration_graph(self, tat: List[str], **graph_kwargs):
        """
        Displays the collaboration graph of the top N authors.

        Args:
            tat (List[str]): A list of authors.
            **graph_kwargs: Other graph parameters.

        Raises:
            TypeError: If `tat` is not a list of strings.
            FileNotFoundError: If the output directory does not exist and cannot be created.
            RuntimeError: If there is an error during the process.
        """

        if not isinstance(tat, list) or not all(isinstance(author, str) for author in tat):
            raise TypeError("Error: 'tat' must be a list of strings.")

        # Ensure the output directory exists
        output_dir = "figures"  # Change to your output directory
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                raise FileNotFoundError(f"Error creating output directory: {e}")

        try:
            bibfile = pbx_probe(
                file_bib=os.path.join("./input", self.matched_csv), db=self.database, del_duplicated=True
            )
            self.save_network_collab_output(
                bibfile.network_collab,
                entry="aut",
                tgt=tat,
                filename=f"{output_dir}/{'_'.join(tat)}.svg",  # Changed the filename
                **graph_kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Error during graph display: {e}")

    def author_collaboration(self, n=12) -> str:
        """
        Gets the top N authors with the highest degree and the title and abstract of their related articles for the specified field.

        Args:
            n (int): The number of authors with the highest degree to return.

        Returns:
            str: A JSON string containing author information.

        Raises:
            TypeError: If `n` is not an integer.
            RuntimeError: If there is an error during the process.
        """
        if not isinstance(n, int):
            raise TypeError("Error: 'n' must be an integer.")

        try:
            pagerank_centrality = nx.pagerank(self.author_graph)

            # Sort by PageRank value in descending order and take the top n authors
            top_authors = sorted(
                pagerank_centrality.items(), key=lambda x: x[1], reverse=True
            )[:n]

            records = []
            authors = []

            for author, pagerank in top_authors:
                author_id = author.author_id
                author_obj = self.manager.get_author_by_id(author_id)

                if author_obj:
                    author_name = f"{author_obj.full_name} ({author_id})"
                    authors.append(author_obj.name.lower())
                    article_contents = "; ".join(
                        [
                            f"article {i + 1}: {article.title}"
                            for i, article in enumerate(author_obj.articles)
                        ]
                    )
                    records.append(
                        {
                            "author": author_name,
                            "content": article_contents,
                            "pagerank_centrality": pagerank,
                        }
                    )
                else:
                    print(f"Warning: Author object with author_id {author_id} not found.")

            self.show_author_collaboration_graph(
                tat=authors,
                rows=4,
                cols=3,
                wspace=0.2,
                hspace=0.2,
                tspace=0.01,
                node_size=300,
                font_size=8,
                pad=0.2,
                nd_a="#FF0000",
                nd_b="#008000",
                nd_c="#808080",
                verbose=False,
            )

            final_df = pd.DataFrame(records).head(n)
            final_json = final_df.to_json(orient="records", indent=4)

            return final_json

        except Exception as e:
            raise RuntimeError(f"Error during author collaboration analysis: {e}")

    def certain_author(self, author: str) -> str:
        """
        Gets detailed information about a specified author, including collaborations and related articles.

        Args:
            author (str): The name or full name of the target author.

        Returns:
            str: A JSON string containing the author's detailed information.

        Raises:
            TypeError: If `author` is not a string.
            RuntimeError: If there is an error during the process.
        """
        if not isinstance(author, str):
            raise TypeError("Error: 'author' must be a string.")

        target = self.manager.get_author_by_name(name_or_full_name=author)

        if target is None:
            return json.dumps(
                {"error": f"Error: Author '{author}' not found."}, ensure_ascii=False
            )

        try:
            self.show_author_collaboration_graph(
                tat=[f"{target.name}".lower()],
                rows=4,
                cols=3,
                wspace=0.2,
                hspace=0.2,
                tspace=0.01,
                node_size=300,
                font_size=8,
                pad=0.2,
                nd_a="#FF0000",
                nd_b="#008000",
                nd_c="#808080",
                verbose=False,
            )
        except Exception as e:
            print(f"Warning: Unable to execute show_author_collaboration_graph: {str(e)}")

        try:
            collaborations = []
            connected_nodes = self.graph_builder.find_connected_nodes(
                self.author_graph, target
            )

            for collaborator, edge_data in connected_nodes:
                coauthored_articles = [
                    article.title
                    for article in target.articles
                    if collaborator.author_id in article.authors
                ]
                collaborations.append(
                    {
                        "name": collaborator.full_name,
                        "coauthored_articles": coauthored_articles,
                    }
                )

            article_to_coauthors = {}
            for collaboration in collaborations:
                author_name = collaboration["name"]
                articles = collaboration["coauthored_articles"]
                for article in articles:
                    if article not in article_to_coauthors:
                        article_to_coauthors[article] = []
                    article_to_coauthors[article].append(author_name)

            collaborations_grouped = [
                {"title": article, "coauthors": authors}
                for article, authors in article_to_coauthors.items()
            ]

            res = {
                "name": target.full_name,
                "institute": target.institute,
                "collaborations": collaborations_grouped,
            }

            res_json = json.dumps(res, ensure_ascii=False, indent=4)
            return (
                f"Following is the supplementary information for the user's query about certain author. "
                f"The information including the target author's name, institution, and collaboration. "
                f"Your are supposed to answer user's query based on the following data\n{res_json}\n"
            )

        except Exception as e:
            raise RuntimeError(f"Error constructing result: {str(e)}")

    def _cosine_similarity(self, query: np.ndarray, embeddings: np.ndarray):
            """
            Calculates the cosine similarity between a query vector and a set of embedding vectors.

            Args:
                query (np.ndarray): The query vector.
                embeddings (np.ndarray): The set of embedding vectors.

            Returns:
                np.ndarray: An array of cosine similarity scores.

            Raises:
                TypeError: If `query` or `embeddings` is not a numpy array.
                ValueError: If `query` or `embeddings` have incompatible shapes for cosine similarity calculation.
            """
            if not isinstance(query, np.ndarray) or not isinstance(embeddings, np.ndarray):
                raise TypeError("Error: Both 'query' and 'embeddings' must be numpy arrays.")

            query_norm = np.linalg.norm(query)
            embeddings_norm = np.linalg.norm(embeddings, axis=1)

            if query_norm == 0:
                raise ValueError("Error: 'query' vector has zero norm.")
            if np.any(embeddings_norm == 0):
                raise ValueError("Error: 'embeddings' contains vectors with zero norm.")

            dot_product = np.dot(embeddings, query.T)
            return dot_product / (query_norm * embeddings_norm)

    def certain_entity(self, name: str) -> str:
        """
        Retrieves information about a specific entity based on its name.

        Args:
            name (str): The name of the entity to search for.

        Returns:
            str: A JSON string containing information about the top 10 most similar entities.

        Raises:
            TypeError: If `name` is not a string.
            RuntimeError: If there is an error during the entity retrieval process.
            FileNotFoundError: If the entity parquet files are not found.
        """
        if not isinstance(name, str):
            raise TypeError("Error: 'name' must be a string.")

        try:
            if not Entity.embedding_client:
                raise ValueError("Error: Embedding client is not initialized in Entity class.")

            entities = Entity.read_name_description_from_parquet()
            new_embedding = np.array(
                get_embedding(text=name, embedding_client=Entity.embedding_client)
            )

            res = []
            for entity in entities:
                score = self._cosine_similarity(new_embedding, entity.name_embedding)[0][0]
                entity_with_score = (entity, score)
                if len(res) < 10:
                    res.append(entity_with_score)
                else:
                    min_score_entity = min(res, key=lambda x: x[1])

                    if score > min_score_entity[1]:
                        res.remove(min_score_entity)
                        res.append(entity_with_score)

            res.sort(key=lambda x: x[1], reverse=True)
            res_entity = [entity for entity, _ in res]
            final_res = []
            for entity in res_entity:
                final_res.append(
                    {"name": entity.name, "description": entity.description}
                )

            return pd.DataFrame(final_res).to_json(orient="records", indent=4)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: Entity file not found: {e}")

        except Exception as e:
            raise RuntimeError(f"Error during entity retrieval: {e}")
        
    def certain_article(self, query: str) -> str:
        '''
            the authorship of certain paper
        '''

        article = self.manager.get_article_by_doi(query)
        if article:
            res = {"title": article.title,
                   "doi": article.doi,
                   "year": article.year,
                   "abstract": article.abstract,
                   "Research Entities": article.entities}
            return json.dumps(res, indent=4)
        
        match_ratio = 0.9
        best_match_article = None   

        for title, articles in self.manager.article_title_dict.items():
            ratio = self._calculate_similarity(query, title)
            if ratio >= match_ratio:
                match_ratio = ratio
                best_match_article = articles[0]  # Get the first matching article
                print(f"Type: {type(best_match_article)}")

        if best_match_article != None: 

            res = {"title": best_match_article.title,
                   "doi": best_match_article.doi,
                   "year": best_match_article.year,
                   "abstract": best_match_article.abstract,
                   "Research Entities": best_match_article.entities}
            
            return json.dumps(res, indent=4)
        else: 
            return "No matched article."