# Installation requirements:
# pip install sentence-transformers faiss-cpu numpy pyppeteer transformers torch pymupdf
# For Lightpanda: Download binary from https://github.com/lightpanda-io/browser/releases or use Docker
# For ACI.dev: pip install aci-sdk (from https://github.com/aipotheosis-labs/aci-python-sdk)
# Start Lightpanda CDP server: ./lightpanda serve --host 127.0.0.1 --port 9222 (or Docker: docker run -d -p 9222:9222 lightpanda/browser:nightly)

import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pyppeteer import connect  # For Lightpanda integration
# from aci_sdk import ACIClient  # ACI.dev Python SDK
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import fitz  # PyMuPDF for PDF parsing
import os

# Step 1: Parse CV from PDF
def parse_pdf_cv(pdf_path):
    """
    Extract text from a PDF CV using PyMuPDF.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Extracted text from the CV.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"Error parsing PDF {pdf_path}: {e}")
        return ""

# Step 2: Semantic Search Initial Ranking with Sentence-BERT + FAISS
def initial_ranking(cvs, job_description):
    """
    Generate embeddings and rank CVs semantically using Sentence-BERT.
    Args:
        cvs (list): List of CV texts.
        job_description (str): Job description text.
    Returns:
        list: List of (CV text, distance) tuples, sorted by ascending distance.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    cv_embeddings = model.encode(cvs, show_progress_bar=True)
    jd_embedding = model.encode([job_description], show_progress_bar=True)

    dimension = cv_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(cv_embeddings)
    
    k = min(5, len(cvs))  # Top 5 or all if fewer
    distances, indices = index.search(jd_embedding, k)
    
    ranked_cvs = [(cvs[i], dist) for i, dist in zip(indices[0], distances[0])]
    return sorted(ranked_cvs, key=lambda x: x[1])  # Sort by ascending distance

# Step 2: Reranking with ZeroEntropy
# Step 5: Reranking with ZeroEntropy (Updated)
def rerank_with_zeroentropy(ranked_cvs, job_description):
    """
    Rerank top CVs using ZeroEntropy cross-encoder for refined scores.
    Args:
        ranked_cvs (list): List of (CV text, distance) tuples from initial ranking.
        job_description (str): Job description text.
    Returns:
        list: List of (CV text, initial distance, rerank score) tuples, sorted by descending score.
    """
    reranker_model_name = "zeroentropy/zerank-1"  # Replace with actual model or API
    tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
    reranker_model.eval()

    scores = []
    for cv, _ in ranked_cvs:
        # Prepare inputs for the model
        inputs = tokenizer(job_description, cv, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        # Get model output
        with torch.no_grad():
            outputs = reranker_model(**inputs)
            logits = outputs.logits  # Shape: (1, 2) for binary classification
            print(f"Logits shape: {logits.shape}, Values: {logits}")
        # Extract the positive relevance score (index 0 for the first class, adjust if model docs specify otherwise)
        relevance_score = torch.sigmoid(logits[:, 0]).item()  # Use the first logit and apply sigmoid
        scores.append(relevance_score)

    # Re-rank by descending score
    reranked = sorted(zip(ranked_cvs, scores), key=lambda x: x[1], reverse=True)
    return [(cv, dist, score) for ((cv, dist), score) in reranked]

# Main Pipeline Execution
async def run_pipeline(job_description, pdf_paths=None, web_cv_url=None, use_aci=False, aci_params=None):
    """
    End-to-end pipeline: Ingest CVs (PDF, web, ACI), rank, and rerank.
    Args:
        job_description (str): Job description text.
        pdf_paths (list): List of paths to PDF CVs.
        web_cv_url (str): URL for web-based CV (optional).
        use_aci (bool): Whether to fetch CVs via ACI.dev.
        aci_params (dict): Parameters for ACI tool call (optional).
    """
    cvs = []
    
    # Integration Point 1: Parse PDF CVs
    if pdf_paths:
        for pdf_path in pdf_paths:
            cv_text = parse_pdf_cv(pdf_path)
            if cv_text:
                cvs.append(cv_text)
                print(f"Parsed CV from {pdf_path}: {cv_text[:100]}...")
    
    # # Integration Point 2: Lightpanda for web scraping
    # if web_cv_url:
    #     scraped_cv = await scrape_cv_with_lightpanda(web_cv_url)
    #     if scraped_cv:
    #         cvs.append(scraped_cv)
    #         print(f"Scraped CV from {web_cv_url}: {scraped_cv[:100]}...")
    
    # # Integration Point 3: ACI.dev for tool-based data fetch
    # if use_aci:
    #     aci = ACIClient(api_key="your_aci_api_key")  # Replace with your API key
    #     additional_cvs = call_aci_tool(aci, "cv_database_query", aci_params or {"query": job_description})
    #     cvs.extend(additional_cvs)
    #     print(f"Fetched {len(additional_cvs)} CVs via ACI.dev tool.")
    
    # Fallback sample CVs if none fetched
    if not cvs:
        print("No CVs provided, using sample CVs.")
        cvs = [
            "Software Engineer with 5 years in Python and Agile project management.",
            "Full-stack Developer, expert in JavaScript and SQL databases.",
            "Data Scientist skilled in ML and data analysis using R."
        ]
    
    # Step 2: Initial semantic ranking
    print("Performing initial semantic ranking...")
    ranked_cvs = initial_ranking(cvs, job_description)
    # Step 3: Reranking
    print("Reranking with ZeroEntropy...")
    final_ranked = rerank_with_zeroentropy(ranked_cvs, job_description)
    
    # Output results
    print("\nFinal Ranked CVs:")
    for rank, (cv, dist, score) in enumerate(final_ranked, 1):
        print(f"Rank {rank}: {cv[:100]}... (Initial Distance: {dist:.4f}, Rerank Score: {score:.4f})")
    
    # Output results and extract top candidate's number
    print("\nFinal Ranked CVs:")
    top_cv, top_dist, top_score = final_ranked[0]  # First ranked CV
    print(f"Rank 1: {top_cv[:100]}... (Initial Distance: {top_dist:.4f}, Rerank Score: {top_score:.4f})")
    
    # Extract phone number from the top CV (simple pattern matching)
    import re
    phone_pattern = r'\+[0-9]{9,15}'  # Matches + followed by 9-15 digits (e.g., +33777036339)
    top_phone = re.search(phone_pattern, top_cv)
    if top_phone:
        top_phone_number = top_phone.group(0)
        print(f"Extracted phone number of top candidate: {top_phone_number}")
    else:
        print("No phone number found in top CV. Using fallback: +33612345678")
        top_phone_number = "+33612345678"  # Fallback number

# Example Usage
if __name__ == "__main__":
    job_desc = "We are seeking a highly skilled Machine Learning Engineer with expertise in computer vision, deep learning, and quantum-inspired algorithms. The ideal candidate will have hands-on experience developing AI models for predictive analysis in semiconductor environments, adapting classical algorithms to quantum frameworks, and evaluating R&D projects. This role involves transitioning proof-of-concepts (POCs) to production, contributing to academic publications, and participating in AI hackathons. The position requires strong proficiency in Python-based ML frameworks and a passion for emerging technologies like blockchain, DeFi, and AI for robotics/drones."
    pdf_paths = []  # Replace with actual PDF paths
    for file in os.listdir("cv"):
        if file.endswith(".pdf"):
            pdf_paths.append(f"cv/{file}")
    # web_url = "https://example.com/sample-cv"  # Replace with actual URL
    # aci_params = {"query": "Python developers"}  # Example ACI tool params
    
    asyncio.run(run_pipeline(
        job_description=job_desc,
        pdf_paths=pdf_paths
    ))