##############################################MindLink AI###############################################

# import streamlit as st
# import spacy
# import fitz  # PyMuPDF
# import networkx as nx
# import plotly.graph_objects as go
# from sentence_transformers import SentenceTransformer, util
# from spacy.lang.en.stop_words import STOP_WORDS
# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry
# import concurrent.futures
# import re

# st.set_page_config(
#     page_title="MindLink AI",
#     page_icon="💡",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# st.markdown(
#     """
#     <style>
#     .block-container {
#         padding-top: 2rem;
#         padding-bottom: 2rem;
#     }
#     .stRadio > div {
#         flex-direction: row;
#     }
#     .stTextArea textarea {
#         font-size: 1rem;
#         line-height: 1.6;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# st.title("💡 MindLink: Learn Smarter, Not Harder")
# st.caption(
#     "Unlock hidden connections between concepts with AI-powered knowledge mapping."
# )


# @st.cache_resource
# def load_model():
#     return spacy.load("en_core_web_sm")


# @st.cache_resource
# def load_embedder():
#     return SentenceTransformer("all-MiniLM-L6-v2")


# nlp = load_model()
# embedder = load_embedder()


# def create_retry_session():
#     session = requests.Session()
#     retry_strategy = Retry(
#         total=3,
#         backoff_factor=0.5,
#         status_forcelist=[429, 500, 502, 503, 504],
#         allowed_methods=frozenset(["GET"]),
#     )
#     adapter = HTTPAdapter(max_retries=retry_strategy)
#     session.mount("http://", adapter)
#     session.mount("https://", adapter)
#     return session


# def identify_core_ideas(text, top_n=20):
#     doc = nlp(text)
#     candidate_phrases = {}
#     for chunk in doc.noun_chunks:
#         phrase = chunk.text.strip()
#         if len(phrase) > 3 and not any(token.is_stop for token in chunk):
#             key = phrase.lower()
#             candidate_phrases[key] = candidate_phrases.get(
#                 key, {"text": phrase, "score": 0}
#             )
#             candidate_phrases[key]["score"] += text.lower().count(
#                 phrase.lower()
#             ) + 0.5 * len(phrase.split())

#     for ent in doc.ents:
#         if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT"}:
#             key = ent.text.lower()
#             candidate_phrases[key] = candidate_phrases.get(
#                 key, {"text": ent.text, "score": 0}
#             )
#             candidate_phrases[key]["score"] += 1.5

#     sorted_phrases = sorted(
#         candidate_phrases.values(), key=lambda x: x["score"], reverse=True
#     )
#     return [p["text"] for p in sorted_phrases[:top_n]]


# class ConceptProcessor:
#     def extract_text_from_pdf(self, uploaded_file):
#         with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
#             return "".join(page.get_text() for page in doc)


# class ConceptEnricher:
#     def __init__(self):
#         self.cache = {}
#         self.session = create_retry_session()

#     def enrich(self, concept):
#         if concept in self.cache:
#             return self.cache[concept]

#         sources = self.fetch_all_sources(concept)
#         definition, examples, source_url = self.extract_information(concept, sources)

#         result = {
#             "concept": concept,
#             "definition": definition,
#             "examples": examples[:3],
#             "url": source_url,
#         }
#         self.cache[concept] = result
#         return result

#     def fetch_all_sources(self, concept):
#         api_calls = [
#             {
#                 "url": "https://api.duckduckgo.com/",
#                 "params": {
#                     "q": concept,
#                     "format": "json",
#                     "no_redirect": 1,
#                     "skip_disambig": 1,
#                 },
#             },
#             {
#                 "url": "https://en.wikipedia.org/w/api.php",
#                 "params": {
#                     "action": "query",
#                     "format": "json",
#                     "titles": concept,
#                     "prop": "extracts",
#                     "exintro": True,
#                     "explaintext": True,
#                 },
#             },
#         ]

#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             futures = [
#                 executor.submit(self.get_api_data, api["url"], api["params"])
#                 for api in api_calls
#             ]
#             return [
#                 f.result()
#                 for f in concurrent.futures.as_completed(futures)
#                 if f.result()
#             ]

#     def extract_information(self, concept, sources):
#         definition, examples, url = "No definition available", [], ""

#         for data in sources:
#             if "Abstract" in data:
#                 abstract = data.get("Abstract", "")
#                 if abstract:
#                     definition = abstract.split(".")[0] + "."
#                     examples += self.extract_examples(abstract)
#                     url = data.get("AbstractURL", url)

#             elif "query" in data:
#                 for page in data.get("query", {}).get("pages", {}).values():
#                     if "extract" in page:
#                         content = page["extract"]
#                         if content:
#                             definition = content.split(".")[0] + "."
#                             examples += self.extract_examples(content)
#                             url = f"https://en.wikipedia.org/wiki/{concept.replace(' ', '_')}"

#         return definition, list(dict.fromkeys(examples)), url

#     def get_api_data(self, url, params):
#         try:
#             resp = self.session.get(url, params=params, timeout=6)
#             return resp.json() if resp.status_code == 200 else None
#         except:
#             return None

#     def extract_examples(self, text):
#         return re.split(r"(?<=\.|\?)\s", text)[:3]


# class LLMConceptConnector:
#     def __init__(self, embedder):
#         self.embedder = embedder
#         self.embeddings = {}

#     def get_embedding(self, phrase):
#         if phrase not in self.embeddings:
#             self.embeddings[phrase] = self.embedder.encode(
#                 phrase, convert_to_tensor=True
#             )
#         return self.embeddings[phrase]

#     def find_similarity(self, a, b):
#         return float(util.pytorch_cos_sim(self.get_embedding(a), self.get_embedding(b)))


# class KnowledgeGraphBuilder:
#     def __init__(self):
#         self.graph = nx.Graph()

#     def build(self, concepts, connector, threshold=0.5):
#         for concept in concepts:
#             self.graph.add_node(concept, size=10)

#         for i, source in enumerate(concepts):
#             for j, target in enumerate(concepts):
#                 if i >= j:
#                     continue
#                 sim = connector.find_similarity(source, target)
#                 if sim >= threshold:
#                     self.graph.add_edge(
#                         source,
#                         target,
#                         weight=sim,
#                         label=f"{source} ↔ {target} ({sim:.2f})",
#                     )

#         if len(self.graph.edges) == 0:
#             return None

#         pos = nx.spring_layout(self.graph, k=0.6)
#         for node in self.graph.nodes:
#             self.graph.nodes[node]["pos"] = pos[node]
#         return self.graph

#     def visualize(self):
#         edge_x, edge_y, edge_text = [], [], []
#         for src, tgt, data in self.graph.edges(data=True):
#             x0, y0 = self.graph.nodes[src]["pos"]
#             x1, y1 = self.graph.nodes[tgt]["pos"]
#             edge_x += [x0, x1, None]
#             edge_y += [y0, y1, None]
#             edge_text.append(data["label"])

#         edge_trace = go.Scatter(
#             x=edge_x,
#             y=edge_y,
#             mode="lines",
#             line=dict(width=1, color="#999"),
#             hoverinfo="text",
#             text=edge_text,
#         )
#         node_x, node_y, labels = [], [], []
#         for node in self.graph.nodes:
#             x, y = self.graph.nodes[node]["pos"]
#             node_x.append(x)
#             node_y.append(y)
#             labels.append(node)

#         node_trace = go.Scatter(
#             x=node_x,
#             y=node_y,
#             mode="markers+text",
#             text=labels,
#             textposition="top center",
#             marker=dict(size=22, color="#00BFFF"),
#             hoverinfo="text",
#         )
#         return go.Figure(
#             data=[edge_trace, node_trace],
#             layout=go.Layout(
#                 showlegend=False,
#                 hovermode="closest",
#                 margin=dict(b=10, l=10, r=10, t=10),
#                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#             ),
#         )


# # ============================ APP FLOW ============================

# processor = ConceptProcessor()
# enricher = ConceptEnricher()

# col1, col2 = st.columns([1, 2])
# with col1:
#     mode = st.radio(
#         "Choose input method:",
#         ["✍️ Text Input", "⬆️ Upload PDF"],
#         index=0,
#     )

# text = ""
# if mode == "✍️ Text Input":
#     text = st.text_area("Enter your study content here:", height=200)
# else:
#     file = st.file_uploader("Upload a PDF", type=["pdf"])
#     if file:
#         text = processor.extract_text_from_pdf(file)
#         st.text_area("Extracted Text", value=text, height=200, disabled=True)

# if text and st.button("🚀 Analyze Concepts"):
#     with st.spinner("🔍 Finding core ideas..."):
#         concepts = identify_core_ideas(text)
#         st.success(f"➡️ {len(concepts)} core ideas found")

#     with st.spinner("🧠 Building knowledge graph..."):
#         connector = LLMConceptConnector(embedder)
#         graph_builder = KnowledgeGraphBuilder()
#         graph = graph_builder.build(concepts, connector)
#         if graph:
#             fig = graph_builder.visualize()
#             st.subheader("🧩 Concept Map")
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.warning("No meaningful connections found between concepts.")

#     with st.spinner("🌐 Enriching concepts with background info..."):
#         enriched = []
#         progress = st.progress(0)
#         for i, concept in enumerate(concepts):
#             enriched.append(enricher.enrich(concept))
#             progress.progress((i + 1) / len(concepts))

#     st.subheader("📘 Concept Details")
#     for data in enriched:
#         with st.expander(data["concept"]):
#             st.markdown(f"**Definition:** {data['definition']}")
#             if data["examples"]:
#                 st.markdown("**Examples:**")
#                 for ex in data["examples"]:
#                     st.markdown(f"- {ex}")
#             if data["url"]:
#                 st.markdown(f"[🌐 Learn More]({data['url']})")


# MindLink App Using Open-Source LLM for Concept Understanding and Graph Building
import streamlit as st
import spacy
import fitz  # PyMuPDF
import networkx as nx
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModel
import torch
from spacy.lang.en.stop_words import STOP_WORDS
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import concurrent.futures
import re

st.set_page_config(
    page_title="MindLink AI",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stRadio > div {
        flex-direction: row;
    }
    .stTextArea textarea {
        font-size: 1rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💡 MindLink: Learn Smarter, Not Harder")
st.caption(
    "Unlock hidden connections between concepts with AI-powered knowledge mapping."
)


@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_llm_embedder():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    return tokenizer, model


nlp = load_model()
tokenizer, model = load_llm_embedder()


def create_retry_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def identify_core_ideas(text, top_n=20):
    doc = nlp(text)
    candidate_phrases = {}
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if len(phrase) > 3 and not any(token.is_stop for token in chunk):
            key = phrase.lower()
            candidate_phrases[key] = candidate_phrases.get(
                key, {"text": phrase, "score": 0}
            )
            candidate_phrases[key]["score"] += text.lower().count(
                phrase.lower()
            ) + 0.5 * len(phrase.split())

    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT"}:
            key = ent.text.lower()
            candidate_phrases[key] = candidate_phrases.get(
                key, {"text": ent.text, "score": 0}
            )
            candidate_phrases[key]["score"] += 1.5

    sorted_phrases = sorted(
        candidate_phrases.values(), key=lambda x: x["score"], reverse=True
    )
    return [p["text"] for p in sorted_phrases[:top_n]]


class ConceptProcessor:
    def extract_text_from_pdf(self, uploaded_file):
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)


class ConceptEnricher:
    def __init__(self):
        self.cache = {}
        self.session = create_retry_session()

    def enrich(self, concept):
        if concept in self.cache:
            return self.cache[concept]

        sources = self.fetch_all_sources(concept)
        definition, examples, source_url = self.extract_information(concept, sources)

        result = {
            "concept": concept,
            "definition": definition,
            "examples": examples[:3],
            "url": source_url,
        }
        self.cache[concept] = result
        return result

    def fetch_all_sources(self, concept):
        api_calls = [
            {
                "url": "https://api.duckduckgo.com/",
                "params": {
                    "q": concept,
                    "format": "json",
                    "no_redirect": 1,
                    "skip_disambig": 1,
                },
            },
            {
                "url": "https://en.wikipedia.org/w/api.php",
                "params": {
                    "action": "query",
                    "format": "json",
                    "titles": concept,
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                },
            },
        ]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.get_api_data, api["url"], api["params"])
                for api in api_calls
            ]
            return [
                f.result()
                for f in concurrent.futures.as_completed(futures)
                if f.result()
            ]

    def extract_information(self, concept, sources):
        definition, examples, url = "No definition available", [], ""

        for data in sources:
            if "Abstract" in data:
                abstract = data.get("Abstract", "")
                if abstract:
                    definition = abstract.split(".")[0] + "."
                    examples += self.extract_examples(abstract)
                    url = data.get("AbstractURL", url)

            elif "query" in data:
                for page in data.get("query", {}).get("pages", {}).values():
                    if "extract" in page:
                        content = page["extract"]
                        if content:
                            definition = content.split(".")[0] + "."
                            examples += self.extract_examples(content)
                            url = f"https://en.wikipedia.org/wiki/{concept.replace(' ', '_')}"

        return definition, list(dict.fromkeys(examples)), url

    def get_api_data(self, url, params):
        try:
            resp = self.session.get(url, params=params, timeout=6)
            return resp.json() if resp.status_code == 200 else None
        except:
            return None

    def extract_examples(self, text):
        return re.split(r"(?<=\.|\?)\s", text)[:3]


class LLMConceptConnector:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.embeddings = {}

    def embed(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            model_output = self.model(**inputs)
        embeddings = model_output.last_hidden_state.mean(dim=1)
        return embeddings

    def get_embedding(self, phrase):
        if phrase not in self.embeddings:
            self.embeddings[phrase] = self.embed(phrase)
        return self.embeddings[phrase]

    def find_similarity(self, a, b):
        v1 = self.get_embedding(a)
        v2 = self.get_embedding(b)
        sim = torch.nn.functional.cosine_similarity(v1, v2).item()
        return sim


class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()

    def build(self, concepts, connector, threshold=0.5):
        for concept in concepts:
            self.graph.add_node(concept, size=10)

        for i, source in enumerate(concepts):
            for j, target in enumerate(concepts):
                if i >= j:
                    continue
                sim = connector.find_similarity(source, target)
                if sim >= threshold:
                    self.graph.add_edge(
                        source,
                        target,
                        weight=sim,
                        label=f"{source} ↔ {target} ({sim:.2f})",
                    )

        if len(self.graph.edges) == 0:
            return None

        pos = nx.spring_layout(self.graph, k=0.6)
        for node in self.graph.nodes:
            self.graph.nodes[node]["pos"] = pos[node]
        return self.graph

    def visualize(self):
        edge_x, edge_y, edge_text = [], [], []
        for src, tgt, data in self.graph.edges(data=True):
            x0, y0 = self.graph.nodes[src]["pos"]
            x1, y1 = self.graph.nodes[tgt]["pos"]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_text.append(data["label"])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="#999"),
            hoverinfo="text",
            text=edge_text,
        )
        node_x, node_y, labels = [], [], []
        for node in self.graph.nodes:
            x, y = self.graph.nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)
            labels.append(node)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=22, color="#00BFFF"),
            hoverinfo="text",
        )
        return go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=10, l=10, r=10, t=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )


processor = ConceptProcessor()
enricher = ConceptEnricher()

col1, col2 = st.columns([1, 2])
with col1:
    mode = st.radio(
        "Choose input method:",
        ["✍️ Text Input", "⬆️ Upload PDF"],
        index=0,
    )

text = ""
if mode == "✍️ Text Input":
    text = st.text_area("Enter your study content here:", height=200)
else:
    file = st.file_uploader("Upload a PDF", type=["pdf"])
    if file:
        text = processor.extract_text_from_pdf(file)
        st.text_area("Extracted Text", value=text, height=200, disabled=True)

if text and st.button("🚀 Analyze Concepts"):
    with st.spinner("🔍 Finding core ideas..."):
        concepts = identify_core_ideas(text)
        st.success(f"➡️ {len(concepts)} core ideas found")

    with st.spinner("🧠 Building knowledge graph..."):
        connector = LLMConceptConnector(tokenizer, model)
        graph_builder = KnowledgeGraphBuilder()
        graph = graph_builder.build(concepts, connector)
        if graph:
            fig = graph_builder.visualize()
            st.subheader("🧩 Concept Map")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No meaningful connections found between concepts.")

    with st.spinner("🌐 Enriching concepts with background info..."):
        enriched = []
        progress = st.progress(0)
        for i, concept in enumerate(concepts):
            enriched.append(enricher.enrich(concept))
            progress.progress((i + 1) / len(concepts))

    st.subheader("📘 Concept Details")
    for data in enriched:
        with st.expander(data["concept"]):
            st.markdown(f"**Definition:** {data['definition']}")
            if data["examples"]:
                st.markdown("**Examples:**")
                for ex in data["examples"]:
                    st.markdown(f"- {ex}")
            if data["url"]:
                st.markdown(f"[🌐 Learn More]({data['url']})")
