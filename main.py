from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
import requests
from sentence_transformers import SentenceTransformer
import time


load_dotenv()


def get_current_weather(city_name, latitude, longitude):
    """
    Fetches current weather data for a given city.
    
    Args:
        city_name (str): Name of the city
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
    
    Returns:
        dict: Weather data containing temperature and wind speed, or None if failed
    """
    url = os.environ.get("WEATHER_URL")
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json().get("current_weather", {})
        return {
            "temperature": data.get("temperature"),
            "wind_speed": data.get("windspeed")
        }
    else:
        return None


def initialize_components(index_name="city-weather-data"):
    """
    Initializes embedding model and sets up Pinecone index.
    
    Args:
        index_name (str): Name of the Pinecone index
    
    Returns:
        tuple: (embedding_model, pinecone_index)
    """
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    if pc.has_index(index_name):
        print(f"Index '{index_name}' already exists. Using existing index.")
    else:
        print(f"Index '{index_name}' does not exist. Creating a new index.")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        time.sleep(30)
    
    index = pc.Index(index_name)
    
    return embedding_model, index


def fetch_and_store_weather_data(cities, embedding_model, index):
    """
    Fetches weather data for all cities and stores it in Pinecone.
    
    Args:
        cities (dict): Dictionary of city names and their coordinates
        embedding_model: Sentence transformer model
        index: Pinecone index object
    
    Returns:
        dict: Weather data for all cities
    """
    weather_data = {}
    city_names = list(cities.keys())
    city_embeddings = embedding_model.encode(city_names)

    for i, (city, (lat, lon)) in enumerate(cities.items()):
        weather_info = get_current_weather(city, lat, lon)
        if weather_info:
            weather_data[city] = weather_info
            index.upsert([
                {
                    "id": city,
                    "values": city_embeddings[i].tolist(),
                    "metadata": {
                        "temperature": weather_info["temperature"],
                        "wind_speed": weather_info["wind_speed"]
                    }
                }
            ])
    
    return weather_data


def query_weather(prompt, embedding_model, index):
    """
    Query function that accepts a prompt containing a city name and returns 
    today's temperature and wind speed information.
    
    Args:
        prompt (str): A text prompt containing a city name
        embedding_model: Sentence transformer model
        index: Pinecone index object
    
    Returns:
        str: Formatted weather information or error message
    """
    prompt_embedding = embedding_model.encode([prompt])
    
    query_results = index.query(
        vector=prompt_embedding[0].tolist(),
        top_k=1,
        include_metadata=True
    )
    
    if not query_results.matches:
        return "Sorry, I couldn't find weather information for any city in your query."
    
    best_match = query_results.matches[0]
    city_name = best_match.id
    metadata = best_match.metadata
    similarity_score = best_match.score
    
    if similarity_score < 0.3:
        return f"I found a possible match for '{city_name}', but I'm not very confident. Could you be more specific?"
    
    temperature = metadata.get('temperature')
    wind_speed = metadata.get('wind_speed')
    
    return f"Today's weather in {city_name}:\n- Temperature: {temperature}°C\n- Wind speed: {wind_speed} km/h"


def run_test_queries(embedding_model, index):
    """
    Runs test queries to demonstrate the weather query functionality.
    
    Args:
        embedding_model: Sentence transformer model
        index: Pinecone index object
    """
    test_queries = [
        "What's the weather in Madrid?",
        "How's Barcelona today?",
        "Temperature in Bilbao",
        "Madrid weather",
        "What about Barcelona?"
    ]
    
    print("\nRunning test queries:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"Query: '{query}'")
        print(query_weather(query, embedding_model, index))
        print("-" * 50)


def main():
    """Main function to orchestrate the entire pipeline."""
    cities = {
        "Madrid": (40.42, -3.70),
        "Barcelona": (41.39, 2.16),
        "Bilbao": (43.26, -2.93)
    }
    
    embedding_model, index = initialize_components()
    
    weather_data = fetch_and_store_weather_data(cities, embedding_model, index)
    
    print("\nWeather data collected:")
    print(weather_data)
    
    run_test_queries(embedding_model, index)


if __name__ == "__main__":
    main()