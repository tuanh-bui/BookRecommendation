"""
Main file to run the project.
"""

import proj2_books_2
import time

b = proj2_books_2.BookData()
b.load_book_data("medium_book_data.csv")
b.load_genre_data()
b.sort_genre_data()
user_lib = proj2_books_2.BookRecommendations()

print("Welcome, what would you like to do? ")
user_input = ""

while True:
    print("\nWhat would you like to do? \n1. See Tree Map\n2. See Bar Graph\n3. See Weighted Graph"
          "\n4. Get Book recommendations\n5. See Read library\n6. Stop")
    user_input = input("\nEnter choice: ").lower().strip()
    if user_input == '6':
        exit()
    elif user_input == '1':
        b.tree_map()
    elif user_input == '2':
        print("\nHow many genres would you like printed in the graph? Maximum is", b.num_genres())
        end = int(input("\nEnter choice: "))
        b.bar_graph(end)
    elif user_input == '4':
        print('\n We would want to take a few data from you to better personalize your book recommendations'
              '\n The output we provide has the title of the book with the highest compability to lowest')
        time.sleep(2)
        title = input('\n Please enter the books you have read (type "stop" when you are done): ')
        while title != 'stop':
            if not user_lib.add_book(title):
                print("Sorry that book is not in our data. ")
            title = input('\n Enter another book (or "stop"): ')
        time.sleep(1)
        r = input('\n Please choose the lowest rating you want (0.0 - 5.0)')
        p = input('\n Please choose the number of pages you want')
        gs = input('\n Please choose the genres you like (type "stop" when done)')
        gen_li = []
        while gs != 'stop':
            gen_li.append(gs)
            gs = input('\n Please choose another genre ( or "stop"): ')
        lim = input('\n Please choose the number of recommendations you want')
        print(user_lib.final_reccomendations(user_lib.sim_score_rating(float(r)), user_lib.sim_score_pages(int(p)),
                                             user_lib.sim_score_genres(gen_li), int(lim)))
    elif user_input == '5':
        if not user_lib.get_book_titles():
            print({'No books added'})
        else:
            print(user_lib.get_book_titles())
    elif user_input == '3':
        b.display_tree()
