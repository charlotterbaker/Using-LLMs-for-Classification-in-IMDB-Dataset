# Why and When to Use LLMs for Classification using IMDB Movie Review Dataset
## Source: CSCE 580 Fall 2025 Project B, Assigned by B. Srivastava

### Objective: 
The goal of this project is to understand the benefits and trade-offs in using different forms of AI. This was done by building and testing different models for sentiment analysis (a) fine-tune a pre-trained transformer model, DistilBERT, on the IMDB movie review dataset, and compare its performance with both its base version (without fine-tuning) and traditional machine learning
algorithms for sentiment classification. 

### Dataset Overview: 
Used the **IMDB Movie Review Dataset:**
(https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-
reviews), which contains **50,000** movie reviews labeled as either positive or
negative

### Brief Overview of Findings
**Status:** Completed, Main Results:
**TF-IDF + Logistic Regression:** 90.0% accuracy
**Fine-tuned DistilBERT:** 90.0% accuracy
**GPT-2 + LR:** 81.9% accuracy
**Base DistilBERT:** 50.0% accuracy
**Experience:**
Began with classical TF-IDF and logistic regression model which was surprisingly strong. Base distilBERT predicted almost everything as positive until fine-tuning. Was able to read the training curves and confusion matrices. Designed small AI testcases to see where the models fail.


Further analyses, graphs, and conclusions on findings are included in Report.md

Assignment written/assigned by B. Srivastava
Completed by Charlotte Baker
