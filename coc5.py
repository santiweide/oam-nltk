import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import networkx as nx
import pandas as pd
from dash import dash_table
import nltk
import base64
import io
import plotly.graph_objects as go

from dash.dash_table.Format import Group
from dash.dependencies import Input, Output, State
from itertools import combinations

from nltk.tokenize import word_tokenize
from flask_caching import Cache
import numpy as np
import spacy
from nltk.corpus import wordnet as wn
from collections import Counter
from sklearn.preprocessing import normalize

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from collections import defaultdict
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get the embeddings for a sentence using BERT
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the last hidden state (embeddings for each token)
    embeddings = outputs.last_hidden_state
    # We average the embeddings of all tokens to get the sentence embedding
    sentence_embedding = embeddings.mean(dim=1).squeeze().numpy()
    return sentence_embedding

# Function to extract lexicon terms from text
def extract_lexicon_terms(text, lexicon_terms):
    tokens = word_tokenize(text.lower())
    return [(token, idx) for idx, token in enumerate(tokens) if token in lexicon_terms]

# Function to compute pairwise cosine similarity between embeddings
def compute_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Learning the co-occurrence matrix based on proximity
def learn_lexicon_embeddings(selected_docs, oam_lexicon):
    # Initialize a dictionary to store the embeddings for each lexicon term
    term_embeddings = defaultdict(list)
    co_occurrence_matrix = np.zeros((len(oam_lexicon), len(oam_lexicon)))

    # List of lexicon terms
    lexicon_terms = list(oam_lexicon.keys())

    # Iterate over the selected documents
    for doc in selected_docs:
        # Get the list of lexicon terms with their positions
        lexicon_term_positions = extract_lexicon_terms(doc, oam_lexicon)
        
        # Get BERT embeddings for the entire document
        doc_embedding = get_bert_embeddings(doc)

        # For each pair of lexicon terms in the document, compute their cosine similarity
        for (term1, idx1), (term2, idx2) in combinations(lexicon_term_positions, 2):
            if term1 in lexicon_terms and term2 in lexicon_terms:
                # Get embeddings for the terms
                embedding1 = get_bert_embeddings(term1)
                embedding2 = get_bert_embeddings(term2)

                # Compute the cosine similarity
                similarity = compute_cosine_similarity(embedding1, embedding2)

                # Adjust similarity based on their proximity (inverse of distance)
                distance = abs(idx1 - idx2)  # Proximity: lower distance means higher similarity
                proximity_factor = 1 / (1 + distance)  # Inverse of distance
                weighted_similarity = similarity * proximity_factor

                # Get indices for the terms in the lexicon
                idx_i = lexicon_terms.index(term1)
                idx_j = lexicon_terms.index(term2)

                # Update the co-occurrence matrix with the weighted similarity
                co_occurrence_matrix[idx_i, idx_j] += weighted_similarity
                co_occurrence_matrix[idx_j, idx_i] += weighted_similarity

    # Normalize the co-occurrence matrix
    if np.max(co_occurrence_matrix) > 0:
        co_occurrence_matrix /= np.max(co_occurrence_matrix)

    # Return the updated co-occurrence matrix
    return co_occurrence_matrix

nltk.download("punkt")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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

options = []
for category, items in lexicon_groups.items():
    options.append({"label": category, "value": category, "disabled": True})  # Group label
    options.extend({"label": f"\u2003 {item}", "value": item} for item in items)  # Indented items

synonym_to_term = {syn: term for term, synonyms in oam_lexicon.items() for syn in synonyms}

documents = []

app.layout = dbc.Container([
    html.H1("Keyword Co-occurrence Graph Analysis"),

    dbc.Row([
        dbc.Col(html.Label("Select Terms from OAM Lexicon"), width=4),
        dbc.Col(dcc.Dropdown(
            id="lexicon-dropdown",
            options=options,
            multi=True,
            placeholder="Select terms..."
        ), width=8),
    ]),
    
    html.Hr(),

    dbc.Row([
        dbc.Col(html.Label("Upload Text Files"), width=4),
        dbc.Col(dcc.Upload(
            id="upload-data",
            children=html.Button("Upload Files"),
            multiple=True
        ), width=8),
    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col(html.Label("Select Documents"), width=4),
        dbc.Col(dcc.Dropdown(
            id="document-dropdown",
            options=[],
            multi=True
        ), width=8),
    ]),

    html.Hr(),
    dcc.Slider(
        id='threshold-slider',
        min=0, max=1, step=0.01, value=0.1,
        marks={0: '0', 0.5: '0.5', 1: '1'},
        tooltip={'placement': 'bottom', 'always_visible': True}
    ),

    dbc.Row([
        dbc.Col(html.Button("Generate Graph", id="generate-btn", n_clicks=0), width=4),
        dbc.Col(dcc.Graph(id="network-graph"), width=8),
    ]),

    html.Hr(),
    html.Div(id="table-container"),  # This ensures the table is included in the layout


    dbc.Row([
        dbc.Col(html.Button("Download Co-occurrence Matrix", id="download-btn", n_clicks=0)),
        dcc.Download(id="download-dataframe-xlsx")
    ]),
])

@app.callback(
    Output("document-dropdown", "options"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def upload_documents(contents, filenames):
    global documents
    if contents:
        for content, filename in zip(contents, filenames):
            content_str = base64.b64decode(content.split(",")[1]).decode("utf-8")
            documents.append((filename, content_str))
    return [{"label": doc[0], "value": i} for i, doc in enumerate(documents)]



# @app.callback(
#     Output("network-graph", "figure"),
#     Output("download-dataframe-xlsx", "data"),
#     Output("table-container", "children"),
#     Input("generate-btn", "n_clicks"),
#     Input("download-btn", "n_clicks"),
#     Input("threshold-slider", "value"),
#     State("document-dropdown", "value"),
#     prevent_initial_call=True
# )
@app.callback(
    Output("network-graph", "figure"),
    Output("table-container", "children"),
    Input("generate-btn", "n_clicks"),
    State("document-dropdown", "value"),
    prevent_initial_call=True
)
def generate_graph(n_clicks_graph, selected_docs):
    if not selected_docs:
        return go.Figure(), html.Div("No data available")

    # Get the texts of the selected documents
    selected_texts = [documents[i][1] for i in selected_docs]

    # Get BERT embeddings for each document
    doc_embeddings = [get_bert_embeddings(doc) for doc in selected_texts]

    # Create a co-occurrence matrix based on cosine similarity of embeddings
    num_docs = len(selected_texts)
    co_occurrence_matrix = learn_lexicon_embeddings(selected_texts, oam_lexicon)

    print(co_occurrence_matrix)

    # Create a graph
    G = nx.Graph()
    for i, doc in enumerate(selected_texts):
        G.add_node(i, label=f"Doc {i + 1}")

    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            weight = co_occurrence_matrix[i, j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    # Define positions for the nodes
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]['label'])

    # Create the edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="gray"),
        hoverinfo="text",
        mode="lines"
    )

    # Create the node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=15, color="lightblue", line=dict(width=2, color="black"))
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Document Co-occurrence Graph",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Create the table for document-co-occurrence mapping
    table_data = [{"Document": f"Doc {i + 1}", "Co-occurrence With": ", ".join([f"Doc {j + 1}" for j in range(num_docs) if i != j and co_occurrence_matrix[i, j] > 0])} for i in range(num_docs)]
    table_component = dash_table.DataTable(
        data=table_data,
        columns=[{"name": col, "id": col} for col in ["Document", "Co-occurrence With"]],
        style_table={'overflowX': 'scroll', 'maxHeight': '500px', 'overflowY': 'auto'},
        style_cell={'textAlign': 'center'},
        page_size=10
    )

    return fig, html.Div([html.H4("Document Co-occurrence Mapping"), table_component])

if __name__ == "__main__":
    app.run_server(debug=True)
