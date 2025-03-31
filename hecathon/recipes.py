import requests

SPOONACULAR_API_KEY = "a30c673d13ff482fab22ac497bc27b8e"


def get_recipes(ingredients):
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    headers = {"x-api-key": SPOONACULAR_API_KEY}
    params = {
        "ingredients": ingredients,
        "number": 3  # Limit to 3 recipes to save API points
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        recipes = response.json()

        # Print basic details
        message = ""

        for recipe in recipes:
            message += f"🍽️ Recipe: {recipe['title']}\n"
            message += f"📸 Image: {recipe['image']}\n"
            message += f"✅ Uses {recipe['usedIngredientCount']} ingredients you have.\n"
            message += f"❌ Missing {recipe['missedIngredientCount']} ingredients.\n\n"

        return message

    else:
        print("⚠️ Error fetching recipes.")


# Test with some ingredients
get_recipes("chicken,pepper,pasta")