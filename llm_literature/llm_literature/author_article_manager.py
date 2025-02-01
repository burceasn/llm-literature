import pandas as pd

class Author:
    def __init__(self, author_id, name, full_name, institute):
        """
        Initialize an Author object.

        Parameters:
        - author_id (str): Unique identifier for the author.
        - name (str): Short version of the author's name.
        - full_name (str): Full name of the author.
        - institute (str): The institution associated with the author.
        """
        self.author_id = author_id
        self.name = name  # Short name from "Author" column
        self.full_name = full_name  # Full name from "Author full names" column
        self.institute = institute or "No Found"
        self.articles = []  # Initialize articles list

    def __str__(self):
        return (f"Author ID: {self.author_id}, Name: {self.name}, Full Name: {self.full_name}, "
                f"Institute: {self.institute}, Articles: {[article.doi for article in self.articles]}")

    def __hash__(self):
        return hash(self.author_id)

    def __eq__(self, other):
        if isinstance(other, Author):
            return self.author_id == other.author_id
        return False

class Article:
    def __init__(self, title, doi, year, abstract, source, reference, authors=None):
        """
        Initialize an Article object.

        Parameters:
        - title (str): Title of the article.
        - doi (str): DOI of the article.
        - year (int): Publication year.
        - abstract (str): Abstract of the article.
        - source (str): Source journal or conference.
        - reference (str): Reference information.
        - authors (list): List of author IDs associated with the article.
        """
        self.title = title
        self.doi = doi
        self.year = year
        self.abstract = abstract
        self.source = source
        self.reference = reference
        self.authors = authors if authors is not None else []
        self.entities = []

    def __str__(self):
        return f"Title: {self.title}, DOI: {self.doi}, Year: {self.year}, Authors: {self.authors}"

class AuthorArticleManager:
    def __init__(self, data):
        """
        Manager to handle Author and Article objects.

        Parameters:
        - data (pd.DataFrame): Preprocessed DataFrame containing article and author information.
        """
        self.data = data
        self.authors = {}  # Dictionary to store Author objects with author_id as keys
        self.articles = {}  # Dictionary to store Article objects with DOI as keys
        self.author_name_dict = {}  # Dictionary to map full names to Author objects
        self.author_short_name_dict = {}
        self.article_title_dict = {}  # Dictionary to map titles to Article objects
        self._process_data()

    def _process_data(self):
        """
        Process data to build authors and articles.
        """
        for _, row in self.data.iterrows():
            author_ids = self._process_authors(row)
            self._process_articles(row, author_ids)

        # After processing all rows, associate articles with their authors
        for article in self.articles.values():
            self._match_author_article(article)

    def _process_authors(self, row):
        """
        Process authors from a row and populate the authors dictionary.

        Parameters:
        - row (pd.Series): A row from the DataFrame.

        Returns:
        - author_ids (list): List of author IDs associated with the article.
        """
        author_full_entries = row['Author full names'].split('; ')
        author_short_entries = row['Author'].split('; ')
        author_ids = []

        if len(author_full_entries) != len(author_short_entries):
            # If the number of authors in 'Author' and 'Author full names' do not match,
            # fallback to using full_name as short_name
            print(f"Warning: Number of authors in 'Author' and 'Author full names' do not match for DOI {row['DOI']}.")

        # Iterate over both short and full author entries in parallel
        for i, full_entry in enumerate(author_full_entries):
            if '(' in full_entry and ')' in full_entry:
                # Extract full_name and author_id
                full_name_part, id_part = full_entry.rsplit('(', 1)
                full_name = full_name_part.strip()
                author_id = id_part.strip(')')

                # Extract short_name from the "Author" column if available
                if i < len(author_short_entries):
                    short_name = author_short_entries[i].strip()
                else:
                    short_name = full_name  # Fallback to full_name if mismatch occurs

                # Get institute using short_name
                institute = self._get_institute(short_name, row['Author with Institution'])

                if author_id in self.authors:
                    author = self.authors[author_id]
                    # 检查现有作者的属性是否与当前条目一致
                    if author.name != short_name or author.full_name != full_name or author.institute != institute:
                        # 按照用户要求，不打印任何信息
                        pass  # 可以选择记录到日志文件中，而不是打印
                else:
                    # Create a new Author object
                    author = Author(author_id, short_name, full_name, institute)
                    self.authors[author_id] = author

                    # Map full_name to Author object
                    if full_name not in self.author_name_dict:
                        self.author_name_dict[full_name] = []
                    self.author_name_dict[full_name].append(author)

                    # **新增**：Map short_name to Author object
                    if short_name not in self.author_short_name_dict:
                        self.author_short_name_dict[short_name] = []
                    self.author_short_name_dict[short_name].append(author)

                
                # Append author_id to the list
                author_ids.append(author_id)
        return author_ids

    def _process_articles(self, row, author_ids):
        """
        Process articles from a row and populate the articles dictionary.

        Parameters:
        - row (pd.Series): A row from the DataFrame.
        - author_ids (list): List of author IDs associated with the article.
        """
        doi = row['DOI']
        if doi in self.articles:
            # Article already exists, append new authors if not already present
            existing_article = self.articles[doi]
            for author_id in author_ids:
                if author_id not in existing_article.authors:
                    existing_article.authors.append(author_id)
        else:
            # Create a new Article object
            article = Article(
                title=row['Title'],
                doi=doi,
                year=row['Year'],
                abstract=row['abstract'],
                source=row['Source'],
                reference=row['reference'],
                authors=author_ids.copy()  # Use copy to prevent accidental modifications
            )
            self.articles[doi] = article

            title_lower = row['Title'].lower()
            if title_lower not in self.article_title_dict:
                self.article_title_dict[title_lower] = []
            self.article_title_dict[title_lower].append(article)

    def _get_institute(self, short_name, author_institute_info):
        """
        Extract the institution corresponding to the given short_name from the author_institute_info string.

        Parameters:
        - short_name (str): The short name of the author.
        - author_institute_info (str): The string containing author names and their institutes.

        Returns:
        - institute (str): The institution associated with the author.
        """
        for entry in author_institute_info.split('; '):
            # Split each entry at the first comma
            parts = entry.split(',', 1)
            if len(parts) != 2:
                continue  # Skip malformed entries
            name, institution = parts
            name = name.strip()
            institution = institution.strip()
            if name == short_name:
                return institution
        return "No Found"

    def _match_author_article(self, article):
        """
        Match an article to its authors by populating the authors' articles lists.

        Parameters:
        - article (Article): The article to be matched with its authors.
        """
        for author_id in article.authors:
            author = self.authors.get(author_id)
            if author:
                if article not in author.articles:
                    author.articles.append(article)

    def show_authors(self):
        """
        Displays all authors' information in a DataFrame format.
        """
        authors_data = []
        for author in self.authors.values():
            authors_data.append({
                'Author ID': author.author_id,
                'Name': author.name,
                'Full Name': author.full_name,
                'Institute': author.institute,
                'Articles': [article.title for article in author.articles]  # Titles of articles
            })
        
        if authors_data:
            df = pd.DataFrame(authors_data)
            return df
        else:
            print("No authors found.")

    def show_articles(self):
        """
        Displays all articles' information in a DataFrame format.
        """
        articles_data = []
        for article in self.articles.values():
            articles_data.append({
                'Title': article.title,
                'DOI': article.doi,
                'Year': article.year,
                'Abstract': article.abstract,
                'Source': article.source,
                'Reference': article.reference,
                'Authors': [self.authors.get(author_id).full_name for author_id in article.authors if self.authors.get(author_id)]  # Full names of authors
            })
        
        if articles_data:
            df = pd.DataFrame(articles_data)
            return df
        else:
            print("No articles found.")

    def get_author_by_id(self, author_id):
        """
        Retrieve an Author object by its ID.

        Parameters:
        - author_id (str): The ID of the author.

        Returns:
        - Author object or None if not found.
        """
        return self.authors.get(author_id)
    
    def get_author_by_name(self, name_or_full_name):
        """
        Retrieve a list of Author objects by their name or full name.
        This function checks both short_name and full_name.

        Parameters:
        - name_or_full_name (str): The name or full name of the author.

        Returns:
        - List of Author objects.
        """
        authors = self.author_name_dict.get(name_or_full_name, [])
        if authors:
            return authors[0]  # Return the first matching Author object

        # Check short name if full name is not found
        short_authors = self.author_short_name_dict.get(name_or_full_name, [])
        if short_authors:
            return short_authors[0]  # Return the first matching Author object

        # If no author is found, return None
        return None

    def get_article_by_doi(self, doi):
        """
        Retrieve an Article object by its DOI.

        Parameters:
        - doi (str): The DOI of the article.

        Returns:
        - Article object or None if not found.
        """
        return self.articles.get(doi)

    def get_article_by_title(self, title):
        """
        Retrieve a list of Article objects by their title.

        Parameters:
        - title (str): The title of the article.

        Returns:
        - List of Article objects.
        """
        return self.article_title_dict.get(title.lower(), [])