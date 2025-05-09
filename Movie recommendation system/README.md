## Movie Recommendation Dashboard<br>

This repo contains a simple and interactive Movie Recommendation System built with Dash and Python. The dashboard allows users to easily search for a movie and get personalized recommendations along with movie covers.


## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Steps for Execution](#steps-for-execution)
- [Selected Approach and its Limitations](#selected-approach-and-its-limitations)
- [Data Source](#data-source)


## Introduction<br>

The recommendation engine employs a hybrid approach, analyzing movie attributes such as tags and genres. By utilizing techniques like TF-IDF vectorization and cosine similarity, the system identifies and suggests movies with similar characteristics to the user's input. To refine the recommendations, sentiment analysis is performed on user tags, filtering out movies with negative feedback and prioritizing movies that reflect positive user experiences. 


## Prerequisites<br>

Before running the application, make sure you have the following installed:

- **Python 3.8** or higher (note that some required Python packages are incompatiable with Python 3.13)

You also need to install the required libraries:

```bash
pip install -r requirements.txt
```

## Steps for Exectuion

Follow this steps to set up the project:<br><br>
1. Clone the repository:
   ```bash
   git clone https://github.com/KatjaParak/ML_movie_recommendation_system.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
    ```bash
   python app.py
   ```
    
The app should be up and running on port 8050

## Selected Approach and its Limitations

Since the collaborative-filtering algorithms struggle with cold-start problem - a scenario where a system has difficulty providing  recommendations for new users -  this method was abandoned in favor of implementing content-based filtering. 

However, the chosen approach also has its own limitations. By relying on features like tags or genres, it becomes higly susceptible to creating a filter bubble and limit the diversity of recommendations (for example. entering 'Pulp Fiction' may results in predominantly Tarantino movies recommendations). To mitigate this, sentiment analysis is applied to the tags, to some extent helping enhance the diversity of recommendations. By analyzing sentiment behind the tags, the system can filter out movies with negative or critical tags, ensuring that only neutral and positive tags remains. 


## Data Source

The MovieLens dataset (ml-latest), generated in July 2023, includes user-generated data spanning from January 1995 to July 2023. The data is stored in following files:<br>
- movies.csv
- tags.csv
- links.csv
- ratings.csv
- genome-scores.csv
- genome-tags.csv

Follow link to download:
http://grouplens.org/datasets/.
