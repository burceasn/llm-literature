from .author_article_manager import AuthorArticleManager
import networkx as nx
import pandas as pd

class GraphBuilder:
    """
    A class for building and analyzing an author collaboration graph based on articles managed by AuthorArticleManager.
    """

    def __init__(self, manager: AuthorArticleManager):
        """
        Initializes the GraphBuilder object.

        Args:
            manager (AuthorArticleManager): An instance of AuthorArticleManager that manages authors and articles.

        Raises:
            TypeError: If 'manager' is not an instance of AuthorArticleManager.
        """
        if not isinstance(manager, AuthorArticleManager):
            raise TypeError("Error: 'manager' must be an instance of AuthorArticleManager.")
        self.manager = manager

    def build_author_collaboration_graph(self):
        """
        Builds the author collaboration graph.

        Returns:
            networkx.Graph: The author collaboration graph, where nodes are Author objects and edge weights represent
                            the number of collaborations, calculated as 1 / (number of authors - 1) for each article.
                            Adds 'degree_weight' to each node's attributes, calculated as 1 / number of authors for each article.
                            Adds 'pagerank_centrality' to each node's attributes, which is the PageRank centrality.
                            Adds 'degree_centrality' to each node's attributes, which is the degree centrality.
                            
        Raises:
            RuntimeError: If there is an error during the graph construction or centrality calculation.
        """
        try:
            graph = nx.Graph()

            # Iterate through all articles
            for doi, article in self.manager.articles.items():
                authors = article.authors  # List of author IDs
                # Convert author IDs to Author objects
                author_objects = [self.manager.get_author_by_id(author_id) for author_id in authors]
                # Filter out authors that may not exist
                author_objects = [author for author in author_objects if author is not None]

                # Calculate the weight of each edge: 1 / (number of authors - 1), if there is only one author, the edge weight is 0
                num_authors = len(author_objects)
                if num_authors > 1:
                    edge_weight = 1 / (num_authors - 1)
                else:
                    edge_weight = 0

                for i in range(len(author_objects)):
                    for j in range(i + 1, len(author_objects)):
                        author1 = author_objects[i]
                        author2 = author_objects[j]
                        if graph.has_edge(author1, author2):
                            graph[author1][author2]["weight"] += edge_weight
                        else:
                            graph.add_edge(author1, author2, weight=edge_weight)

                # Add degree_weight to each author for each article, calculated as 1 / number of authors
                for author in author_objects:
                    if author not in graph:
                        graph.add_node(author)  # Ensure the node exists
                    if "degree_weight" not in graph.nodes[author]:
                        graph.nodes[author]["degree_weight"] = 0
                    graph.nodes[author]["degree_weight"] += 1 / num_authors

            # Calculate PageRank centrality
            pagerank_centrality = nx.pagerank(graph)
            nx.set_node_attributes(graph, pagerank_centrality, "pagerank_centrality")

            # Calculate degree centrality
            degree_centrality = dict(graph.degree())
            nx.set_node_attributes(graph, degree_centrality, "degree_centrality")

            return graph
        except Exception as e:
            raise RuntimeError(f"Error during graph construction: {e}")

    def find_connected_nodes(self, graph, node):
        """
        Finds all nodes connected to a given node in the graph.

        Args:
            graph (networkx.Graph): The author collaboration graph.
            node: The node (Author object) to find connections for.

        Returns:
            list: A list of tuples, where each tuple contains a connected Author object and the edge attributes.

        Raises:
            TypeError: If 'graph' is not a networkx.Graph or 'node' is not present in the graph.
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError("Error: 'graph' must be an instance of networkx.Graph.")
        if node not in graph:
            raise ValueError("Error: 'node' is not present in the graph.")

        connected_nodes = []
        for neighbor, edge_data in graph[node].items():
            connected_nodes.append((neighbor, edge_data))
        return connected_nodes

    def get_nodes_dataframe(self, graph):
        """
        Gets a DataFrame containing information about the nodes in the graph, including the author's name, author_id, and weighted degree.

        Args:
            graph (networkx.Graph): The author collaboration graph.

        Returns:
            pd.DataFrame: A DataFrame with columns 'name', 'author_id', and 'degree', sorted by degree in descending order.
        
        Raises:
            TypeError: If 'graph' is not a networkx.Graph.
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError("Error: 'graph' must be an instance of networkx.Graph.")

        data = []
        # Use the calculated weighted degree
        degrees = {node: graph.nodes[node].get("degree_weight", 0) for node in graph.nodes()}

        for node in graph.nodes():
            data.append(
                {
                    "name": node.name,
                    "author_id": node.author_id,
                    "degree": degrees[node],  # Use the weighted degree
                }
            )
        df_nodes = pd.DataFrame(data)
        df_nodes = df_nodes.sort_values(by="degree", ascending=False).reset_index(drop=True)
        return df_nodes

    def get_edges_dataframe(self, graph):
        """
        Gets a DataFrame containing information about the edges in the graph, including the names and author_ids of the two authors and the weight of the edge.

        Args:
            graph (networkx.Graph): The author collaboration graph.

        Returns:
            pd.DataFrame: A DataFrame with columns 'author1_name', 'author1_id', 'author2_name', 'author2_id', and 'weight',
                          sorted by weight in descending order.

        Raises:
            TypeError: If 'graph' is not a networkx.Graph.
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError("Error: 'graph' must be an instance of networkx.Graph.")

        data = []
        for u, v, attrs in graph.edges(data=True):
            data.append(
                {
                    "author1_name": u.name,
                    "author1_id": u.author_id,
                    "author2_name": v.name,
                    "author2_id": v.author_id,
                    "weight": attrs.get("weight", 0),
                }
            )
        df_edges = pd.DataFrame(data)
        df_edges = df_edges.sort_values(by="weight", ascending=False).reset_index(drop=True)
        return df_edges
    
