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


def create_radar_chart_group(tfidf_array, terms, selected_docs):
    print(f"debug radar graph, tfidf_array.shape={tfidf_array.shape}, select_docs={selected_docs}")

    summed_values = np.sum(tfidf_array[selected_docs], axis=0)
    
# Another choice Normalize the summed values using Min-Max scaling
#     min_val = np.min(summed_values)
#     max_val = np.max(summed_values)
#     normalized_values = (summed_values - min_val) / (max_val - min_val)

    # Apply log transformation to summed_values to mitigate extreme values
    log_summed = np.log(summed_values + 1)  # Add 1 to avoid log(0)
    min_log = np.min(log_summed)
    max_log = np.max(log_summed)
    
    # Handle case where all log values are the same to avoid division by zero
    if max_log == min_log:
        normalized_values = np.full_like(log_summed, 0.5)
    else:
        normalized_values = (log_summed - min_log) / (max_log - min_log)
    
    angles = np.linspace(0, 2 * np.pi, len(terms), endpoint=False).tolist()
    
    # Append the first value to make the radar chart "close"
    normalized_values = np.concatenate((normalized_values, [normalized_values[0]]))
    angles += angles[:1]

    fig = go.Figure(go.Scatterpolar(
        r=normalized_values,
        theta=terms.tolist(),
        fill='toself',
        name='',
        line=dict(color='blue')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])  # Maintain [0,1] range
        ),
        showlegend=False,
        title="Radar Chart for Log-Normalized TF-IDF Scores",
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


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

    dbc.Row([
        dbc.Col(html.Label("Rader Chart for Selected Docuemnts"), width=4),
        dcc.Graph(id='radar-chart', style={'height': '60vh'}),
    ]),

    html.Hr(),


    dbc.Row([
        # dbc.Col(html.Label("Co-occurence Network Edge Threshold"), width=4),
        dcc.Slider(
            id='threshold-slider',
            min=0, max=1, step=0.01, value=0.1,
            marks={0: '0', 0.5: '0.5', 1: '1'},
            tooltip={'placement': 'bottom', 'always_visible': True}
        ),
        dbc.Col(html.Button("Generate Graph", id="generate-btn", n_clicks=0), width=4),
    ]),

    dbc.Row([
        # dbc.Col(html.Label("Co-occurence Network Graph"), width=4),
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
            print(f"debug [upload_documents] {filename}")
            content_str = base64.b64decode(content.split(",")[1]).decode("utf-8")
            documents.append((filename, content_str))
    options = [{"label": doc[0], "value": i} for i, doc in enumerate(documents)]
    options.append({"label": "Select All", "value": "all"})
    return options

# New callback to handle "Select All" logic
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
    Output("network-graph", "figure"),
    Output("download-dataframe-xlsx", "data"),
    Output("table-container", "children"),
    Output("radar-chart", "figure"), 
    Input("generate-btn", "n_clicks"),
    Input("download-btn", "n_clicks"),
    Input("threshold-slider", "value"),
    Input("document-dropdown", "value"),
    prevent_initial_call=True
)
def generate_graph(n_clicks_graph, n_clicks_download, threshold, selected_docs):
    if not selected_docs:
        return go.Figure(), None, html.Div("No data available"), go.Figure()

    selected_texts = [documents[i][1] for i in selected_docs]

    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        return " ".join([synonym_to_term[token] for token in tokens if token in synonym_to_term])

    processed_docs = [preprocess_text(doc) for doc in selected_texts]

    # Calculate TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_docs)
    terms = vectorizer.get_feature_names_out()

    tfidf_array = tfidf_matrix.toarray()
    tfidf_df = pd.DataFrame(tfidf_array, columns=terms)
    # Export the TF-IDF and Co-occurrence matrices to Excel
    with pd.ExcelWriter('tfidf_matrix.xlsx', engine='openpyxl') as writer:
        tfidf_df.to_excel(writer, sheet_name='TF-IDF Matrix')

    # Calculate Co-occurrence Matrix
    co_occurrence_matrix = np.zeros((len(terms), len(terms)))
    for doc_vector in tfidf_matrix.toarray():
        indices = np.where(doc_vector > 0)[0]
        for i, j in combinations(indices, 2):
            co_occurrence_matrix[i, j] += 1
            co_occurrence_matrix[j, i] += 1

    # Normalization
    if np.max(co_occurrence_matrix) > 0:
        co_occurrence_matrix /= np.max(co_occurrence_matrix)

    # Apply threshold
    co_matrix_df = pd.DataFrame(np.where(co_occurrence_matrix >= threshold, co_occurrence_matrix, 0),
                                 index=terms, columns=terms)
    
    lexicon_embeddings = {term: [] for term in terms}
    for i, doc in enumerate(processed_docs):
        for term in terms:
            if term in doc:
                lexicon_embeddings[term].append(f"Document {selected_docs[i]}")
    
    embedding_df = pd.DataFrame([(term, ", ".join(docs)) for term, docs in lexicon_embeddings.items()],
                                 columns=["oam_lexicon", "Associated Documents"])
    
    # **Display the matrix and document embedding table on the website**
    table_data = co_matrix_df.reset_index().to_dict("records")
    table_columns = [{"name": col, "id": col} for col in co_matrix_df.reset_index().columns]

    table_component = dash_table.DataTable(
        data=table_data,
        columns=table_columns,
        style_table={'overflowX': 'scroll', 'maxHeight': '500px', 'overflowY': 'auto'},
        style_cell={'textAlign': 'center'},
        page_size=10
    )
    
    embedding_table = dash_table.DataTable(
        data=embedding_df.to_dict("records"),
        columns=[{"name": col, "id": col} for col in embedding_df.columns],
        style_table={'overflowX': 'scroll', 'maxHeight': '500px', 'overflowY': 'auto'},
        style_cell={'textAlign': 'center'},
        page_size=10
    )
    
    combined_table = html.Div([
        html.H4("Co-occurrence Matrix"),
        table_component,
        html.H4("Document Embedding Table"),
        embedding_table
    ])
    
    # Check if download button is triggered
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"].split(".")[0] == "download-btn":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            co_matrix_df.to_excel(writer, sheet_name="Co-occurrence Matrix")
            embedding_df.to_excel(writer, sheet_name="Document Embeddings")
        output.seek(0)
        return go.Figure(), dcc.send_bytes(output.getvalue(), filename="co_occurrence_matrix.xlsx"), combined_table
    
    # Build Network Graph
    G = nx.Graph()
    for term in terms:
        G.add_node(term)

    edges = []
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            weight = co_occurrence_matrix[i, j]
            if weight >= threshold:
                G.add_edge(terms[i], terms[j], weight=weight)
                edges.append((terms[i], terms[j], weight))

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y, edge_text = [], [], []
    for edge in edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{edge[0]} ↔ {edge[1]} ({edge[2]:.2f})")

    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="gray"),
        hoverinfo="text",
        mode="lines",
        text=edge_text
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(size=15, color="lightblue", line=dict(width=2, color="black"))
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title='',
        # title="Keyword Co-occurrence Graph",
        showlegend=False,
        hovermode="closest",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # **Create Radar Chart (New Addition)**
    # print(f"debug tfidf_array={tfidf_array}, type={type(tfidf_array)}, shape={tfidf_array.shape}")
    radar_chart = create_radar_chart_group(tfidf_array, terms, selected_docs)  # Using the first document for illustration

    return fig, None, combined_table, radar_chart

if __name__ == "__main__":
    app.run_server(debug=True)
