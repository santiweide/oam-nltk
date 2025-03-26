from collections import Counter
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from flask_caching import Cache
from itertools import combinations
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import base64
import dash
import dash_bootstrap_components as dbc
from dash import dash_table
import io
import math
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spacy
nltk.download("punkt")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

import numpy as np
import plotly.graph_objects as go

def extend_with_synonyms(base_terms):
    expanded = set(base_terms)
    for term in base_terms:
        for syn in wn.synsets(term):
            for lemma in syn.lemmas():
                word = lemma.name().replace("_", " ")
                if word != term and len(word) > 2:  # Avoid very short words
                    expanded.add(word)
    return list(expanded)

oam_lexicon = {
    "women": extend_with_synonyms(["women", "female", "girl", "lady"]),
    "youth": extend_with_synonyms(["youth"]),
    "children": extend_with_synonyms(["children", "boy", "gril"]),
    "farmers": extend_with_synonyms(["farmers"]),
    "pastoralists": extend_with_synonyms(["pastoralists"]),
    "fishers": extend_with_synonyms(["fishers", "fisherman", "fisherwomen"]),
    "cross-border traders": extend_with_synonyms(["cross-border traders"]),
    "displaced populations": extend_with_synonyms(["displaced populations"]),
    "refugees": extend_with_synonyms(["refugees"]),
    "MSMEs": extend_with_synonyms(["Micro enterprises", "small enterprises", "medium-sized enterprises", "MSME"]),
    
    "rural areas": extend_with_synonyms(["rural areas"]),
    "urban areas": extend_with_synonyms(["urban areas"]),
    "cross-border areas": extend_with_synonyms(["cross-border areas"]),
    "high risk": extend_with_synonyms(["high risk"]),
    "low risk": extend_with_synonyms(["low risk"]),
    "conflict affected": extend_with_synonyms(["conflict affected"]),
    
    "governance": extend_with_synonyms(["governance"]),
    "jobs": extend_with_synonyms(["jobs"]),
    "livelihoods": extend_with_synonyms(["livelihoods"]),
    "digitalization": extend_with_synonyms(["digitalization"]),
    "education": extend_with_synonyms(["education"]),
    "food security": extend_with_synonyms(["food security"]),
    "health": extend_with_synonyms(["health"]),
    "biodiversity": extend_with_synonyms(["biodiversity"]),
    "climate change": extend_with_synonyms(["climate change"]),
    "energy transitions": extend_with_synonyms(["energy transitions"]),
    "infrastructure": extend_with_synonyms(["infrastructure"]),
    
    "state capacity": extend_with_synonyms(["state capacity"]),
    "debt sustainability": extend_with_synonyms(["debt sustainability"]),
    "implementation": extend_with_synonyms(["implementation"]),
    "partnerships": extend_with_synonyms(["partnerships"]),
    "income groups": extend_with_synonyms(["income groups"]),
    "local governance": extend_with_synonyms(["local governance"])
}

# Grouped options
lexicon_groups = {
    "Demographic Groups": ["women", "youth", "children", "farmers", "pastoralists", "fishers", "cross-border traders", "displaced populations", "refugees", "MSMEs"],
    "Geographic Areas": ["rural areas", "urban areas", "cross-border areas", "high risk", "low risk", "conflict affected"],
    "Thematic Areas": ["governance", "jobs", "livelihoods", "digitalization", "education", "food security", "health", "biodiversity", "climate change", "energy transitions", "infrastructure"],
    "Ownership & Partnerships": ["state capacity", "debt sustainability", "implementation", "partnerships", "income groups", "local governance"]
}

synonym_to_term = {syn: term for term, synonyms in oam_lexicon.items() for syn in synonyms}

terms = list(oam_lexicon.keys())
documents = []

app.layout = dbc.Container([
    html.H1("Keyword Co-occurrence Analysis"),
    
    # File upload section
    dbc.Row([
        dbc.Col(html.Label("Upload Text Files"), width=4),
        dbc.Col(dcc.Upload(
            id="upload-data",
            children=html.Button("Upload Files"),
            multiple=True
        ), width=8),
    ]),
    
    html.Hr(),

    # Document selection
    dbc.Row([
        dbc.Col(html.Label("Select Documents"), width=4),
        dbc.Col(dcc.Dropdown(
            id="document-dropdown",
            options=[],
            multi=True
        ), width=8),
    ]),
    
    html.Hr(),
    # Window size selection
    dbc.Row([
        dbc.Col(html.Label("Sliding Window Size"), width=4),
        dbc.Col(dcc.Slider(
            id='window-size-slider',
            min=50, max=200, step=50, value=100,
            marks={i: str(i*50) for i in range(1, 4)},
            tooltip={'placement': 'bottom', 'always_visible': True}
        ), width=8),
    ]),
    
    # Matrix display
    dbc.Row([
        dbc.Col(dash_table.DataTable(
            id='matrix-table',
            style_cell={'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},
            style_table={'overflowX': 'auto'}
        ), width=12)
    ])
])

@app.callback(
    Output("document-dropdown", "options"),
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def upload_documents(contents, filenames):
    global documents
    if contents:
        documents = []
        for content, filename in zip(contents, filenames):
            content_str = base64.b64decode(content.split(",")[1]).decode("utf-8")
            documents.append((filename, content_str))
    options = [{"label": doc[0], "value": i} for i, doc in enumerate(documents)]
    options.append({"label": "Select All", "value": "all"})
    return options

@app.callback(
    Output("document-dropdown", "value"),
    Input("document-dropdown", "value"),
    State("document-dropdown", "options"),
    prevent_initial_call=True
)
def select_all_documents(selected_values, options):
    if selected_values and "all" in selected_values:
        # Extract all document indices (excluding "all")
        doc_indices = [opt["value"] for opt in options if opt["value"] != "all"]
        return doc_indices
    return selected_values


@app.callback(
    Output("matrix-table", "data"),
    Output("matrix-table", "columns"),
    Input("document-dropdown", "value"),
    Input("window-size-slider", "value"),
    prevent_initial_call=True
)
def update_matrix(selected_docs, window_size):
    if not selected_docs or not documents:
        return [], []
    
    print(terms)
    matrix = np.zeros((len(terms), len(terms)))
    term_to_index = {term: i for i, term in enumerate(terms)}
    print(f"debug synonym_to_term={synonym_to_term}\n length of the selected docs={len(selected_docs)}")
    # Process documents
    print()
    for doc_idx in selected_docs:
        text = documents[doc_idx][1].lower()
        tokens = nltk.word_tokenize(text)

        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i + window_size]
            unique_terms = set(window)
            for term1, term2 in combinations(unique_terms, 2):
                if term1 in synonym_to_term and term2 in synonym_to_term:
                    idx1 = term_to_index[synonym_to_term[term1]]
                    idx2 = term_to_index[synonym_to_term[term2]]
                    matrix[idx1, idx2] += 1
                    matrix[idx2, idx1] += 1 
    print(matrix)
    data = []
    for row in matrix:
        data.append({terms[i]: row[i] for i in range(len(terms))})

    columns = [{"name": term, "id": term} for term in terms]

    return data, columns

if __name__ == "__main__":
    app.run_server(debug=True)