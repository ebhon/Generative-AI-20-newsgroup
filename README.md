# Newsgroup-generator

# Text Generation and Analysis with GPT-2

## Overview

This project explores the power of text generation and analysis using GPT-2, a pre-trained model from OpenAI. It focuses on analyzing text from various newsgroups, extracting insights, performing text preprocessing, and utilizing Named Entity Recognition (NER) to identify key entities like people, organizations, and locations. Finally, the project demonstrates how GPT-2 can generate coherent and relevant text based on different topics from a dataset, offering an exciting glimpse into the capabilities of generative AI.

## Table of Contents

- [Project Goals](#project-goals)
- [Dataset](#dataset)
- [Text Preprocessing](#text-preprocessing)
- [Named Entity Recognition](#named-entity-recognition)
- [GPT-2 Model](#gpt-2-model)
- [Results](#results)
- [Pros and Cons](#pros-and-cons)
- [Getting Started](#getting-started)
- [Future Work](#future-work)

## Project Goals

The goal of this project is to:
1. Perform exploratory data analysis (EDA) on text data from various newsgroups.
2. Clean and preprocess the text data for NLP tasks.
3. Apply Named Entity Recognition (NER) to extract valuable entities from the text.
4. Use the GPT-2 model to generate text based on prompts from different topics.
5. Analyze the performance and utility of the GPT-2 model in generating text.

## Dataset

This project uses the **20 Newsgroups dataset**, a collection of approximately 20,000 newsgroup documents, categorized into 20 different topics. These topics span various fields, such as:
- Technology (e.g., computers, graphics)
- Science (e.g., space, medicine)
- Politics
- Religion
- Sports
- Miscellaneous topics

### Key Columns in the Dataset:
- **Text**: The content of the newsgroup posts.
- **Category**: The category/topic of the post (e.g., `rec.autos`, `comp.sys.mac.hardware`).

## Text Preprocessing

### Steps Taken:
- **Stopwords Removal**: Common English words were removed using NLTK’s stopwords corpus.
- **Custom Stopwords**: Additional custom stopwords were filtered out based on dataset-specific terms.
- **Text Cleaning**: Applied regex-based cleaning to remove unwanted characters, redundant characters (e.g., "looooong" -> "long"), and non-alphabetical symbols.
  
These preprocessing steps ensured that the data was in a clean and usable format for analysis and modeling.

## Named Entity Recognition (NER)

Using the **dbmdz/bert-large-cased-finetuned-con1103-english** pre-trained model, NER was performed to identify key entities such as:
- **PER** (People)
- **ORG** (Organizations)
- **LOC** (Locations)
- **MISC** (Miscellaneous)

### Insights from NER:
- **Miscellaneous (MISC)** entities were the most frequent, indicating a broad range of topics.
- **Organizations (ORG)** and **Persons (PER)** were also prevalent, reflecting the focus on business and individual references.
- **Locations (LOC)** appeared the least, suggesting that the dataset didn't focus much on geographical contexts.

## GPT-2 Model

To generate new text based on the patterns in the dataset, we used **GPT-2**, a large transformer-based model trained on a wide variety of internet text. 

### How GPT-2 Was Used:
1. **Load the Dataset**: Text data was fetched from the `fetch_20newsgroups` function.
2. **Text Preparation**: Cleaned and formatted the text data for input to GPT-2.
3. **Text Generation**: GPT-2 generated responses based on specific prompts related to various topics like sports, technology, and politics.

### Example Prompts Used:
- **"Artificial intelligence trends"**
- **"Sports updates"**
- **"Political debate on gun control"**

## Results

The GPT-2 model generated relevant and coherent text, maintaining topic relevance for the most part. However, it sometimes showed repetitiveness and inconsistencies in the generated content. This behavior is typical of generative models, especially without fine-tuning.

### Generated Text Example:
- **Prompt**: *"Artificial intelligence trends"*
- **Generated Text**: "AI is rapidly transforming industries, from healthcare to finance. Machine learning models are now being used to..."

### Sentiment Distribution:
The sentiment analysis of the dataset reveals a strong **negative sentiment bias**, likely reflecting the tone of discussions in certain categories such as politics and religion.

## Pros and Cons

### Pros:
- **Creative Text Generation**: GPT-2 produced diverse and original content based on prompts.
- **Topic Relevance**: The model maintained a good understanding of the prompts and generated coherent responses.
- **Quick and Easy**: Using a pretrained model saved time and effort compared to training a model from scratch.
- **Scalable**: The model can generate content across various topics, making it versatile.

### Cons:
- **Repetitiveness**: Sometimes, the model repeated phrases or sentences.
- **Inconsistencies in Quality**: Some generated text was coherent, while other parts felt off-topic.
- **Limited Fine-Tuning**: The model's performance could be improved with domain-specific fine-tuning.
- **Potential Inaccuracies**: The text sometimes contained inaccuracies, especially on complex topics.

## Getting Started

### Prerequisites:
- Python 3.x
- Libraries: `torch`, `transformers`, `nltk`, `sklearn`, `matplotlib`, `seaborn`

### Installation:
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/text-generation-analysis.git
    cd text-generation-analysis
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the script:
    ```bash
    python main.py
    ```

## Future Work

- **Fine-Tuning GPT-2**: Fine-tuning the model on a specific subset of the dataset could enhance its text generation quality.
- **Sentiment Analysis**: Further analysis of sentiment in different categories of the dataset could yield valuable insights.
- **Topic Modeling**: Implementing topic modeling techniques like LDA (Latent Dirichlet Allocation) to further explore hidden topics in the text.


---

Feel free to explore and contribute! If you have any questions, open an issue or submit a pull request.
