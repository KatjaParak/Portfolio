from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import nltk
import spacy
from utils import normalize_titles, cos_similarity, similarity_df, get_recommendations, extract_features, vectorizer, remove_stopwords_title


nltk.data.path.append("./nltk_data")
nlp = spacy.load("en_core_web_sm")
movies_df = extract_features()
movies_df = normalize_titles(movies_df)
tfidf_matrix = vectorizer(movies_df)
similarity_score = cos_similarity(tfidf_matrix)
sim_df = similarity_df(similarity_score,movies_df)
filtered_titles = remove_stopwords_title(sim_df)


app = Dash(__name__,external_stylesheets=[dbc.themes.QUARTZ])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.H2("Welcome to the movie recommendation engine!", style={"textAlign": "center"}))],  justify="center"),
    dbc.Row([
        dbc.Col(
            dcc.Input(id="movie-title", type="text",
                  placeholder="Please enter a movie title ", style={"width": "75vh", "height":"3vh"}), width="auto"),
        dbc.Col(
            html.Button("Search", id="search-button", n_clicks=0, style={"height":"3vh", "border":"none", "text-align":"center", "opacity": "0.6"}), width="auto")
], justify="center"),
    dbc.Row([
        dbc.Col(
              dcc.Loading(id="loading", type="circle",children=[html.Div([html.Div(id="movie-recommendation")])], overlay_style={"visibility": "visible", "filter":"blur(5px)"}),
            width=12
        )
    ], justify="center")
], style={"marginTop": "25vh"}, fluid=True)


@callback(
    Output("movie-recommendation", "children"),
    Input("search-button", "n_clicks"),
    State("movie-title", "value"),
    prevent_initial_call = True
)
def update_recommendations(n_clicks, movie_name):
   def remove_stopwords_name(movie_name):
            if not movie_name:
                  return ''
            doc = nlp(movie_name)
            return ' '.join([token.text.lower() for token in doc if not token.is_stop])
   
   filtered_movie_name = remove_stopwords_name(movie_name)
   recommendations, covers = get_recommendations(filtered_movie_name,sim_df, similarity_score, movies_df, filtered_titles)

   if n_clicks > 0:
        if not movie_name :
                return [dbc.Row([
                dbc.Col(
                    html.P("Please enter a movie title!", style={"textAlign": "center"}), width="auto"
                )
            ], justify="center")]
   

        if recommendations is None:
            return [dbc.Row([
                dbc.Col(
                    html.P(f"{movie_name} was not found", style={"textAlign": "center"}), width="auto"
                )
            ], justify="center")]

        
        return [dbc.Row([
                dbc.Col([
                html.P(rec,style={"marginTop": "3vh", "whiteSpace": "nowrap", "textAlign": "center", "overflow": "hidden", "textOverflow": "ellipsis"}),
                html.Img(src=cover, style={"marginTop": "5vh", "width":"40%", "height": "70%","textAlign": "center", "display": "block", "margin":"0 auto" }),
                
            ], style={"width": "150px"}) for rec, cover in zip(recommendations, covers)],justify="center",style={})]


if __name__ == "__main__":
    app.run(port=8050,debug=True)