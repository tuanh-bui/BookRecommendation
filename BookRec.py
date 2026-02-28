"""
CSC111 - Project 2: Genres of the most popular books of all time and recommendations.
"""

from __future__ import annotations
from typing import Any, Optional
import csv
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt


class Tree:
    """
    A recursive tree data structure.
.

    Representation Invariants:
        - self._root is not None or self._subtrees == []
        - all(not subtree.is_empty() for subtree in self._subtrees)

    # Private Instance Attributes:
    #   - _root:
    #       The item stored at this tree's root, or None if the tree is empty.
    #   - _subtrees:
    #       The list of subtrees of this tree. This attribute is empty when
    #       self._root is None (representing an empty tree). However, this attribute
    #       may be empty when self._root is not None, which represents a tree consisting
    #       of just one item.
    """

    _root: Optional[Any]
    _subtrees: list[Tree]

    def __init__(self, root: Optional[Any], subtrees: list[Tree]) -> None:
        """Initialize a new Tree with the given root value and subtrees.

                If root is None, the tree is empty.

                Preconditions:
                    - root is not none or subtrees == []
        """
        self._root = root
        self._subtrees = subtrees

    def is_empty(self) -> bool:
        """
        Return if the tree is empty.
        """
        return self._root is None

    def print_items(self) -> None:
        """
        Print all the itemes in the Tree.
        """
        if self.is_empty():
            print("Done")
        else:
            print(self._root)
            for subtree in self._subtrees:
                subtree.print_items()

    def add_subtree(self, subtree: Tree) -> None:
        """
        Add the given list of Trees to the Tree's subtrees.
        """

        self._subtrees.append(subtree)

    def get_subtrees(self) -> list[Tree]:
        """
        Return the Tree's subtrees.
        """

        return self._subtrees

    def get_root(self) -> Any:
        """
        Return the root of the tree.
        """

        return self._root


class BookData:
    """
    Stores the BookData as dictionaries

    Instance attributes:
        - book_data: The mapping of the book title to the list of its corresponding genres and rating
        - genre_data: The mapping of the genre to its occurance of all books

    Representation invariants:
        - [len(self._book_data[item]) = 2 for items in self._book_data]
        - [self.genre_data[value] >=1 for value in self.genre_data]
    """

    book_data: Tree
    genre_data: dict[str, int]
    sorted_genres: list[tuple[str, int]]
    top_genres: Tree

    def __init__(self) -> None:
        self.book_data = Tree("Book Data", [])
        self.genre_data = {}
        self.sorted_genres = []

    def load_book_data(self, filename: str) -> None:
        """
        Read and load the data from import csv file into self.book_data.
        """
        with open(filename) as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                try:
                    subtree = Tree(row[0], [Tree([float(row[3]), self.format_genres(row[7]), int(row[11])], [])])
                    self.book_data.add_subtree(subtree)
                except:
                    pass

    def format_genres(self, s: str) -> list[str]:
        """
        Format the string of genres into a list.
        """
        s = s[1:len(s) - 1]
        lst = s.rsplit(", ")

        for i in range(len(lst)):
            lst[i] = lst[i][1: len(lst[i]) - 1]
            if lst[i] == '':
                lst.pop(i)

        return lst

    def load_genre_data(self) -> None:
        """
        Load information into self.genre_data
        """

        for subtree in self.book_data.get_subtrees():
            for genre in subtree.get_subtrees()[0].get_root()[1]:
                if genre in self.genre_data:
                    self.genre_data[genre] += 1
                else:
                    self.genre_data[genre] = 1

    def sort_genre_data(self) -> None:
        """
        Sort the genres into a list of descending order based on how many books are classified under that genre.
        """

        data = self.genre_data.copy()

        for _ in range(len(self.genre_data)):
            max_genre = self.get_max_genre(data)
            data.pop(max_genre)
            self.sorted_genres.append((max_genre, self.genre_data[max_genre]))

    def get_max_genre(self, data: dict[str, int]) -> str:
        """
        Return the genre with the greater number of books from the given data dictionary.
        """

        max_so_far = list(data.keys())[0]

        for genre in data:
            if data[genre] > data[max_so_far]:
                max_so_far = genre

        return max_so_far

    def bar_graph(self, num: int) -> None:
        """
        Open and display a bar graph comparing genres on the x-axis with the number of books in that
        genre on the y-axis. The given parameter, end, is how many genres the graph will show, going
        from most popular to least popular.

        Preconditions:
            - 0 < end <= len(self.genre_data)
        """

        genre_dict = {'genres': [genre[0] for genre in self.sorted_genres[:num]],
                      'numbooks': [genre[1] for genre in self.sorted_genres[:num]]}

        fig = px.bar(genre_dict, x='genres', y='numbooks')
        fig.show()

    def tree_map(self) -> None:
        """
        Open and display a tree map showing the top ten most popular genres and ten books from each of those genres.
        """
        genres_and_books = self.genre_to_books()

        genre_dict = {'genres': [genre[0] for genre in self.sorted_genres[:10]],
                      'numbooks': [genre[1] for genre in self.sorted_genres[:10]],
                      'books': list(genres_and_books.values())[:10]}

        fig = px.treemap(genre_dict, path=[px.Constant("Book Genre Data"), "genres", "books"], values='numbooks')

        fig.update_traces(root_color='lightgrey')
        fig.update_layout(uniformtext={'minsize': 10, 'mode': 'show'}, margin={'t': 50, 'l': 25, 'r': 25, 'b': 25})

        fig.show()

    def genre_to_books(self) -> dict[str, str]:
        """
        Return a dictionary mapping each of the top ten genres to ten books in that genre.
        """

        genres_to_books = {}
        genres = [data[0] for data in self.sorted_genres[:10]]
        for genre in genres:
            genres_to_books[genre] = ""

        for book in self.book_data.get_subtrees():
            for genre in book.get_subtrees()[0].get_root()[1]:
                if genre in genres_to_books:
                    numbooks = len(genres_to_books[genre].split(", "))
                    if numbooks <= 10:
                        genres_to_books[genre] += book.get_root() + ", <br>"

        for genre in genres_to_books:
            genres_to_books[genre] = genres_to_books[genre][:len(genres_to_books[genre]) - 6]
        return genres_to_books

    def display_tree(self) -> None:
        """
        Open and display a tree representing the ten most popular genres. The weight on the edge from the root to
        each genre represents how many books are in that genre.
        """
        g = nx.Graph()
        top_genres = []

        for data in self.sorted_genres[:10]:
            genre, numbooks = data
            top_genres.append(genre)
            g.add_edge("Top Genres", genre, weight=numbooks)

        pos = {"Top Genres": (0, 0), top_genres[0]: (0, 1.5), top_genres[1]: (9.5, 1.4), top_genres[2]: (15.5, 0.6),
               top_genres[3]: (15.8, -0.4), top_genres[4]: (9.6, -1.4), top_genres[5]: (0, -1.5),
               top_genres[6]: (-8.2, -1.4), top_genres[7]: (-13.3, -0.6), top_genres[8]: (-13.1, 0.4),
               top_genres[9]: (-9.3, 1.3)}

        nx.draw_networkx_edges(g, pos, width=6)
        edge_labels = nx.get_edge_attributes(g, "weight")
        nx.draw_networkx_edge_labels(g, pos, edge_labels)

        nx.draw_networkx_nodes(g, pos, node_size=12000)
        nx.draw_networkx_labels(g, pos, font_size=12, font_family="sans-serif")

        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.show(block=True)

    def num_genres(self) -> int:
        """
        Return the number of genres in the data.
        """

        return len(self.genre_data)


class BookRecommendations:
    """
    A Book recommendation object

    Instance attributes:
        - read: The set of titles of books that the user has read

    # Private instance attribute:
    #    - _book_data: the BookData object from the read set of the user (because we do not want to display everything
    on the screen)

    Representation invariants:
        - [isinstancce(title, str) for title in self.read]
        - len(self._book_data.book_data) == 0 and len(self._book_data.genre_data) == 0 if len(self.read) == 0
    """

    read: set[BookData]
    _book_data: BookData

    def __init__(self) -> None:
        """Initialize the set of read books and the private attribute"""
        self.read = set()
        self._book_data = BookData()

    def __repr__(self) -> str:
        if not self.read:
            return "BookRecommendations(set())"
        book_data = []
        for data in self.read:
            book_info = f"{repr(data)}"
            book_data.append(book_info)
        return f"BookRecommendations({', '.join(book_data)})"

    def load_book_data(self, filename: str) -> None:
        """
        Load information from the csv file to a BookData object
        """
        self._book_data.load_book_data(filename)
        self._book_data.load_genre_data()

    def add_book(self, t: str) -> bool:
        """
        Add the name of the input book into self.read
        At the same time load the information the data of the book into self._book_data.
        If title does not exist in the library, return True if they have successfully added the data.
        Otherwise, add nothing and return False.
        """
        self.load_book_data('medium_book_data.csv')
        book_titles = [tree.get_root().lower() for tree in self._book_data.book_data.get_subtrees()]
        if t.lower() in book_titles:
            for subtree in self._book_data.book_data.get_subtrees():
                if subtree.get_root().lower() == t.lower():
                    book_info = subtree.get_subtrees()[0].get_root()
                    user_book = BookData()  # Original initializer (no title argument)
                    user_book.book_data = Tree(t, [Tree(book_info, [])])  # Set the title correctly
                    self.read.add(user_book)
                    return True
        return False

    def get_user_book(self) -> set[BookData]:
        """Return the mapping of the books the user has read to their BookData objects"""
        return self.read

    def get_book_titles(self) -> set[str]:
        """
        Return the set of the books the user read
        """
        # libr = set()
        # for book in self.read:
        #     for t in book.book_data.get_subtrees():
        #         libr.add(t.get_root())
        # return libr
        libr = set()
        for book in self.read:
            if book.book_data.get_root() is not None:
                libr.add(book.book_data.get_root())
        return libr

    def cal_book_ratings(self, threshold: float) -> float:
        """
        Calculate the averge ratings of all the books the user has read above the float threshold

        Preconditons:
            - 0.0 <= threshold <= 5.0
        """
        num = 0.0
        total = 0.0
        if len(self.read) == 0:
            return num
        for book_data in self.read:
            for subtree in book_data.book_data.get_subtrees():
                rating = subtree.get_subtrees()[0].get_root()[0]
                if rating >= threshold:
                    num += rating
                    total += 1
        if total == 0.0:
            return 0.0

        return num / total

    def sim_score_rating(self, rating: float) -> dict[str, float]:
        """
        Get a BookRecommendations object based on the input rating using self.cal_book_ratings(rating) as a helper
        with priorities sorted

        Precondition:
            - 0 <= rating <= 5.0
            - {len(subtree.get_subtress) > 0 for subtree in self._book_data.book_data.get_subtrees()}
            - {isinstance(subtree, list) for subtree in self._book_data.book_data.get_subtrees()}

        """
        recommendations = {}
        visited = set()

        for subtree in self._book_data.book_data.get_subtrees():
            t = subtree.get_root()
            data = subtree.get_subtrees()[0].get_root()
            book_rating = data[0]
            if book_rating is not None:
                diff = abs(rating - book_rating)
                is_read = t in self.get_book_titles()

                if not is_read and t not in visited:
                    if diff == 0:
                        recommendations[t] = 10
                    else:
                        recommendations[t] = round(1 / diff, 2)
                    visited.add(t)
        return recommendations

    def sim_score_pages(self, max_page: int) -> dict[str, float]:
        """
        Get a BookRecommendations object based on maximum number of pages input

        Preconditions:
            - {len(subtree.get_subtress) > 0 for subtree in self._book_data.book_data.get_subtrees()}
            - {isinstance(subtree, list) for subtree in self._book_data.book_data.get_subtrees()}
        """
        recommendations = {}
        visited = set()

        for subtree in self._book_data.book_data.get_subtrees():
            t = subtree.get_root()
            data = subtree.get_subtrees()[0].get_root()
            book_page = data[2]
            if book_page is not None:
                diff = abs(max_page - book_page)
                is_read = t in self.get_book_titles()

                if not is_read and t not in visited:
                    if diff == 0:
                        recommendations[t] = 10
                    else:
                        recommendations[t] = round(1 / diff, 2)
                    visited.add(t)
        return recommendations

    def sim_score_genres(self, genres: list[str]) -> dict[str, float]:
        """
        Get a BookRecommendations object based on the list of input genres from the user

        Preconditions:
            - {len(subtree.get_subtress) > 0 for subtree in self._book_data.book_data.get_subtrees()}
            - {isinstance(subtree, list) for subtree in self._book_data.book_data.get_subtrees()}

        """
        recommendations = {}
        visited = set()

        for subtree in self._book_data.book_data.get_subtrees():
            t = subtree.get_root()
            data = subtree.get_subtrees()[0].get_root()
            book_genres = data[1]

            intersection = len(set(genres).intersection(book_genres))
            union = len(set(genres).union(set(book_genres)))

            if union > 0:
                similarity = intersection / union
            else:
                similarity = 0.0
            is_read = t in self.get_book_titles()

            if not is_read and t not in visited:
                recommendations[t] = similarity
                visited.add(t)
        return recommendations

    def final_reccomendations(self, rating_score: dict[str, float], page_score: dict[str, float],
                              genres_score: dict[str, float], limit: int) -> list[tuple[str, float]]:
        """
        Returns the list of books with their score of compatibility with the 3 requirements the user provide
        within the limit number also provided by the user
        """
        genre_set = set(genres_score.keys())
        rating_set = set(rating_score.keys())
        pages_set = set(page_score.keys())

        not_read = genre_set.intersection(rating_set, pages_set) - self.get_book_titles()

        rec = []

        for t in not_read:
            if t in rating_score and t in page_score and t in genres_score:
                average = (rating_score[t] + page_score[t] + genres_score[t]) / 3
                rec.append((t, round(average, 2)))
        rec.sort(key=lambda x: x[1], reverse=True)
        return rec[:limit]


if __name__ == '__main__':
    import python_ta

    python_ta.check_all(config={
        'max-line-length': 120,
        'disable': ['R1705', 'E9998', 'E9999'],
        'max-nested-blocks': 10
    })
