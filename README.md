# Duplicate Question Detection Model

This project is aimed at detecting duplicate question pairs using a machine learning model. The model utilizes features such as common words, token features, length-based features, fuzzy matching, and a bag-of-words representation to classify question pairs as either duplicate or not.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)

## Introduction

The model is built to classify pairs of questions as duplicates or not, based on several features extracted from the text of the questions. It uses a `RandomForestClassifier` for model training, and the features are engineered using multiple text analysis techniques.

## Dataset

The dataset used for training the model is a subset of a larger question dataset, containing 50,000 pairs of questions. These pairs are labeled as either duplicate or non-duplicate.

- **Number of samples**: 50,000 question pairs
- **Labels**: `1` for duplicate questions, `0` for non-duplicate questions.

## Preprocessing

Before feeding the data into the model, several preprocessing steps are performed on the questions:

1. **Text Lowercasing**: All text is converted to lowercase.
2. **Removing Special Characters**: Characters such as `%`, `$`, `₹`, `@`, and others are replaced with appropriate string equivalents.
3. **Contraction Expansion**: Common English contractions (e.g., `isn't` to `is not`) are expanded.
4. **HTML Tag Removal**: Any HTML tags are stripped using BeautifulSoup.
5. **Removing Punctuation**: Non-word characters are removed.
6. **Tokenization**: The questions are split into individual tokens (words).

## Feature Extraction

Several features are extracted from the question pairs to capture various aspects of the text that might indicate whether two questions are duplicates:

### 1. Common Words:
- Counts how many words are common between both questions.

### 2. Total Words:
- Computes the total number of unique words in both questions.

### 3. Token Features:
- Various token-based features are extracted, such as common non-stop words, common stop words, and common tokens between the two questions.

### 4. Length Features:
- Features related to the length of the questions, including the absolute difference in length, the average token length, and the longest common substring.

### 5. Fuzzy Matching Features:
- Uses fuzzy string matching techniques to compute:
  - Fuzz ratio
  - Partial ratio
  - Token sort ratio
  - Token set ratio

### 6. Bag of Words (BoW):
- A bag-of-words representation is generated for both questions using `CountVectorizer`. Only the top 3000 features (unigrams and bigrams) are considered.

### Feature Vector:
- The final feature vector is constructed by combining all of the extracted features, including both the statistical and token-based features.

## Model Training

The model is trained using a `RandomForestClassifier` from scikit-learn. Here's how the model is trained:

1. **Vectorization**: The `CountVectorizer` is used to convert the question text into numerical representations. The `max_features` is set to 3000, and the `ngram_range` is set to (1, 2) to include both unigrams and bigrams.
   
2. **Random Forest Classifier**: A `RandomForestClassifier` is trained on the feature vectors. This classifier is well-suited for this task, as it can handle high-dimensional feature spaces effectively.

3. **Model Evaluation**: The model is evaluated using accuracy and confusion matrix metrics to assess its performance.

