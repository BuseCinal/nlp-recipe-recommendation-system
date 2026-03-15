import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load the recipe dataset
file_name = 'recipes.csv'

if not os.path.exists(file_name):
    print(f"Error: '{file_name}' not found.")
    exit()

df = pd.read_csv(file_name)

# Convert ingredients to numerical vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Ingredients'])

# Calculate similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

def recommend_recipe(recipe_name, data, sim_matrix, top_n=2):
    idx = data[data['Recipe_Name'] == recipe_name].index[0]
    
    # Get similarity scores and sort them
    scores = sorted(list(enumerate(sim_matrix[idx])), key=lambda x: x[1], reverse=True)
    
    similar_indices = [i[0] for i in scores[1:top_n+1]]
    
    print(f"\nRecommendations for people who like '{recipe_name}':")
    for i in similar_indices:
        print(f"- {data['Recipe_Name'].iloc[i]}")

# Interactive menu
if __name__ == "__main__":
    print("\n🍽️ Welcome to the AI Recipe Recommender! 🍽️")
    print("Available Recipes:", ", ".join(df['Recipe_Name'].tolist()))

    while True:
        user_input = input("\nWhich recipe would you like to find similar ones for? (Type 'q' to quit): ")
        
        if user_input.lower() == 'q':
            print("Exiting system. Bon appétit!\n")
            break
            
        if user_input in df['Recipe_Name'].values:
            recommend_recipe(user_input, df, similarity_matrix)
        else:
            print(f"Sorry, '{user_input}' is not in our database. Please type an exact name.")
