# 🧠 MindLink: Learn Smarter, Not Harder

**MindLink** is an AI-powered Streamlit app that transforms dense study material into a visual **knowledge graph**, helping you uncover and understand the **core concepts** and how they're connected — using a fully **open-source LLM**.

![MindLink Screenshot](./demo.png) <!-- Optional: add your own screenshot -->

---

## 🚀 What It Does

- ✍️ Accepts **raw text** or **PDF files** as input
- 🧠 Uses an **open-source LLM (`intfloat/e5-small-v2`)** for semantic understanding
- 🔗 Builds an intelligent **concept map** using contextual similarity
- 🌐 Enriches each concept with definitions, examples, and URLs from Wikipedia + DuckDuckGo
- 📊 Displays an **interactive knowledge graph** with Plotly

---

## 🌟 Why MindLink?

Most study tools help you summarize. **MindLink helps you internalize**.

By visually mapping relationships between ideas, MindLink enhances comprehension, retention, and discovery. Whether you're a student, researcher, or lifelong learner — MindLink helps you study smarter, not harder.

---

## 🧱 Built With

| Tech | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io/) | Web UI |
| [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) | PDF parsing |
| [SpaCy](https://spacy.io/) | NLP for noun phrase extraction |
| [Hugging Face Transformers](https://huggingface.co/) | LLM embeddings (`intfloat/e5-small-v2`) |
| [NetworkX](https://networkx.org/) | Graph building |
| [Plotly](https://plotly.com/python/) | Graph visualization |
| [Wikipedia & DuckDuckGo APIs](https://www.mediawiki.org/wiki/API:Main_page) | Concept enrichment |

---

## 🖼 Features at a Glance

- ✅ Text or PDF upload
- ✅ Automatic core idea extraction
- ✅ Context-aware concept connection
- ✅ Concept enrichment with definitions + examples
- ✅ Interactive concept graph visualization
- ✅ 100% open-source, no API keys required

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/mindlink
cd mindlink
streamlit run app.py

