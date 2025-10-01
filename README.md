# Semantic Spotter - Retrieval-Augmented Generation case study for MS - AI/ML
> 📑 RAG System for Insurance Policies — LangChain vs LlamaIndex

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
Semantic Spotter project builds a Retrieval-Augmented Generation (RAG) system for answering natural language questions from insurance policy PDFs.
Implement both LangChain and LlamaIndex pipelines, evaluate them, and compare accuracy, latency, and provenance.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Technologies Used
*Core Python utilities:*
- os, time, json, Path → File handling, paths, runtime utilities
- getpass → Secure API key input
- typing (List, Dict) → Type hinting for clean code

*Data Analysis & Visualization:*
- pandas → For structured tabular evaluation and results storage
- matplotlib.pyplot → To create a range of plots and visualizations

*Colab integration:*
- google.colab.userdata and google.colab.files → For secure key storage and PDF uploads

*LangChain components (used for the first RAG pipeline):*
- PyPDFLoader → Extract text from PDF documents
- RecursiveCharacterTextSplitter → Split text into semantic chunks
- OpenAIEmbeddings → Generate vector embeddings of text chunks
- FAISS → Store and search embeddings
- OpenAI → LLM integration with OpenAI GPT
- RetrievalQA → End-to-end retrieval + answer generation chain

*LlamaIndex components (used for the second RAG pipeline):*
- SimpleDirectoryReader → Load documents from directory
- VectorStoreIndex → Build and manage vector indexes
- Settings → Configure LlamaIndex global settings
- OpenAIEmbedding → OpenAI embeddings integration
- OpenAI (aliased as LlamaOpenAI) → OpenAI LLM integration in LlamaIndex

## Features
- Upload and process insurance policy PDFs (tested on Principal-Sample-Life-Insurance-Policy.pdf)
- Build dual RAG pipelines:
  - LangChain → enterprise-ready, modular
  - LlamaIndex → quick prototyping, lightweight
- Auto-generate evaluation QA dataset from document section headers
- Benchmark on accuracy, latency, provenance

## Conclusions
- Both systems achieved equal accuracy (75%)
- LangChain was faster (~1144 ms vs ~1281 ms)
- Both produced full provenance (100%), which is critical for regulated domains

| System      | Accuracy (%) | Latency (ms) | Provenance (%) |
|-------------|--------------|--------------|----------------|
| LangChain   | 75.0         | 1144.6       | 100.0          |
| LlamaIndex  | 75.0         | 1281.2       | 100.0          |

- Both frameworks are capable of answering insurance policy queries
- LangChain outperforms LlamaIndex in latency while maintaining accuracy and provenance
- For production insurance Q&A systems, LangChain is the recommended choice

## Acknowledgements
I want to credit upGrad for the Master of Science in Machine Learning and Artificial Intelligence (AI/ML) degree alongside IIIT-Bangalore, and LJMU, UK
- This project was inspired by all the Professors who trained us during the Retrieval-Augmented Generation (RAG) system
  
## Contact
Created by [@rajaravisekara] - feel free to contact me, Raja - Sr Architect - AI Cloud

<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
