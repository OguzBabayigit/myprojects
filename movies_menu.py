import matplotlib.pyplot as plt
from difflib import get_close_matches
from colorama import Fore, Style, init

init()

def list_movies(movies):
    print(f'{len(movies)} movies in total')
    for movie, value in movies.items():
        print(f'{movie}: {value}{Style.RESET_ALL}')

movies = {
    "The Shawshank Redemption": 9.5,
    "Pulp Fiction": 9.5,
    "The Room": 3.6,
    "The Godfather": 9.2,
    "The Godfather: Part II": 9.0,
    "The Dark Knight": 9.0,
    "12 Angry Men": 8.9,
    "Everything Everywhere All At Once": 8.9,
    "Forrest Gump": 8.8,
    "Star Wars: Episode V": 8.7
}

def add_movie(movies):
    try:
        new_movie = input(Fore.GREEN + "Enter new movie name: " + Style.RESET_ALL).strip()
        if not new_movie:
            raise ValueError("Movie name cannot be empty.")

        new_movie_rating = float(input(Fore.GREEN + "Enter new movie rating (0-10): " + Style.RESET_ALL).strip())
        if new_movie_rating < 0 or new_movie_rating > 10:
            raise ValueError("Rating must be between 0 and 10.")

        movies[new_movie] = new_movie_rating
        print(Fore.GREEN + f"Movie '{new_movie}' successfully added." + Style.RESET_ALL)
    except ValueError as e:
        print(Fore.RED + f"Invalid access! {e}" + Style.RESET_ALL)

def delete_movie(movies):
    try:
        del_movie = input(Fore.GREEN + "Enter movie name to delete: " + Style.RESET_ALL).strip()
        if not del_movie or del_movie not in movies:
            raise ValueError(f"Movie '{del_movie}' not found in the list.")

        movies.pop(del_movie)
        print(Fore.GREEN + f"Movie '{del_movie}' successfully deleted." + Style.RESET_ALL)
    except ValueError as e:
        print(Fore.RED + f"Invalid access! {e}" + Style.RESET_ALL)

def update_movie(movies):
    try:
        movie_name = input(Fore.GREEN + "Enter movie name: " + Style.RESET_ALL).strip()
        if movie_name not in movies:
            raise ValueError(f"Movie '{movie_name}' not found in the list.")

        updating_rating = float(input(Fore.GREEN + "Enter new movie rating (0-10): " + Style.RESET_ALL).strip())
        if updating_rating < 0 or updating_rating > 10:
            raise ValueError("Rating must be between 0 and 10.")

        movies[movie_name] = updating_rating
        print(Fore.GREEN + f"Movie '{movie_name}' rating successfully updated to {updating_rating}." + Style.RESET_ALL)
    except ValueError as e:
        print(Fore.RED + f"Invalid access! {e}" + Style.RESET_ALL)

def stats(movies):
    average_rating = sum(movies.values()) / len(movies)

    ratings = sorted(movies.values())
    n = len(ratings)
    if n % 2 == 1:
        median_rating = ratings[n // 2]
    else:
        median_rating = (ratings[n // 2 - 1] + ratings[n // 2]) / 2

    max_rating = max(movies.values())
    min_rating = min(movies.values())

    best_movies = [movie for movie, rating in movies.items() if rating == max_rating]
    worst_movies = [movie for movie, rating in movies.items() if rating == min_rating]

    results = {
        "Average Rating": average_rating,
        "Median Rating": median_rating,
        "Best Movies": best_movies,
        "Worst Movies": worst_movies
    }

    for key, value in results.items():
        print(f"{key}: {value}{Style.RESET_ALL}")

def random(movies):
    import random
    random_movie = random.choice(list(movies.keys()))
    random_rating = movies[random_movie]
    print(Fore.GREEN + f"Your movie for tonight: {random_movie}, it's rated {random_rating}" + Style.RESET_ALL)

def search_movie(movies):
    keyword = input(Fore.GREEN + "Enter part of movie name: " + Style.RESET_ALL).strip()
    matches = get_close_matches(keyword, movies.keys(), n=1, cutoff=0.3)
    if matches:
        best_match = matches[0]
        print(Fore.YELLOW + f"The movie '{keyword}' does not exist. Did you mean:\n" + Style.RESET_ALL)
        print(Fore.GREEN + f"{best_match}, {movies[best_match]}" + Style.RESET_ALL)
    else:
        print(Fore.RED + "No movies found matching your search." + Style.RESET_ALL)

def movies_sort(movies):
    sorted_movies = sorted(movies.items(), key=lambda item: item[1], reverse=True)
    for movie, rating in sorted_movies:
        print(f"{movie}, {rating}{Style.RESET_ALL}")

def create_rating_histogram(movies):
    ratings = list(movies.values())
    file_name = input("Enter the filename to save the histogram (e.g., 'histogram.png'): ").strip()
    plt.hist(ratings, bins=10, edgecolor='black', alpha=0.7)
    plt.title("Movie Ratings Histogram")
    plt.xlabel("Ratings")
    plt.ylabel("Frequency")
    plt.savefig(file_name)
    print(Fore.GREEN + f"Histogram saved as '{file_name}'." + Style.RESET_ALL)
    plt.show()

def main():
    while True:
        menu = [
            "Menu:",
            "1. List movies",
            "2. Add movie",
            "3. Delete movie",
            "4. Update movie",
            "5. Stats",
            "6. Random movie",
            "7. Search movie",
            "8. Movies sorted by rating",
            "9. Create Rating Histogram"
        ]
        for i in menu:
            print(Fore.YELLOW + i + Style.RESET_ALL)

        choice = int(input(Fore.GREEN + "\nEnter choice (1-9): ..." + Style.RESET_ALL))
        if choice == 1:
            print("-----------------------------------------------")
            list_movies(movies)
        elif choice == 2:
            add_movie(movies)
        elif choice == 3:
            delete_movie(movies)
        elif choice == 4:
            update_movie(movies)
        elif choice == 5:
            stats(movies)
        elif choice == 6:
            random(movies)
        elif choice == 7:
            search_movie(movies)
        elif choice == 8:
            movies_sort(movies)
        elif choice == 9:
            create_rating_histogram(movies)
        else:
            print(Fore.RED + "Invalid access!" + Style.RESET_ALL)
        print('')
        input(Fore.GREEN + "Press Enter to continue..." + Style.RESET_ALL)

if __name__ == "__main__":
    main()
