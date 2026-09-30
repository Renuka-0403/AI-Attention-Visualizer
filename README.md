# AI Attention Visualizer

AI Attention Visualizer is a beginner-friendly Streamlit web application that extracts text from study-note images using OCR, converts the extracted words into numerical embeddings, and applies a basic scaled dot-product attention mechanism to visualize relative attention scores.

The project demonstrates how text can move through different stages of an AI pipeline, from an image to OCR text, embeddings, and attention visualization.

## Project Overview

The application allows users to upload an image containing study notes. The image is processed using Tesseract OCR to extract the text. The extracted words are then converted into 384-dimensional embeddings using the `all-MiniLM-L6-v2` model.

A basic attention mechanism is applied to these embeddings using Query, Key, and Value projections. The resulting attention scores are normalized and displayed as progress bars, allowing users to visually compare the calculated attention assigned to each word.

## Features

* Upload study-note images in JPG, JPEG, or PNG format
* Extract text from images using Tesseract OCR
* Clean and process extracted words
* Generate 384-dimensional word embeddings
* Apply scaled dot-product attention
* Calculate relative attention scores for words
* Display attention scores using visual progress bars
* Identify the word with the highest calculated attention
* Simple and clean Streamlit interface

## Project Workflow

```text
Study Notes Image
        |
        v
    Tesseract OCR
        |
        v
   Extracted Text
        |
        v
   Word Extraction
        |
        v
 Sentence Transformer
   384-D Embeddings
        |
        v
    Q / K / V
        |
        v
Scaled Dot-Product Attention
        |
        v
 Attention Scores
        |
        v
 Word Attention Visualization
```

## Technologies Used

* Python
* Streamlit
* Tesseract OCR
* pytesseract
* Pillow
* NumPy
* Sentence Transformers
* all-MiniLM-L6-v2

## How It Works

### 1. Image Upload

The user uploads a study-notes image through the Streamlit interface.

Supported formats:

* JPG
* JPEG
* PNG

### 2. OCR Text Extraction

Tesseract OCR processes the uploaded image and extracts the visible text.

The project uses `pytesseract` to connect Python with the Tesseract OCR engine.

### 3. Word Processing

The extracted text is split into individual words.

Basic punctuation is removed, and very short words are filtered out. The application processes a maximum of 20 words for the attention visualization.

### 4. Word Embeddings

Each extracted word is passed through the `all-MiniLM-L6-v2` Sentence Transformer model.

The model produces a 384-dimensional numerical representation for each input word.

These vectors allow the words to be represented mathematically and used for the attention calculation.

### 5. Query, Key, and Value

The embedding matrix is transformed into Query, Key, and Value representations.

The project uses randomly initialized projection matrices for this demonstration.

```text
Embeddings
    |
    +----> Query
    |
    +----> Key
    |
    +----> Value
```

### 6. Scaled Dot-Product Attention

The Query and Key matrices are multiplied to calculate attention scores.

The scores are scaled using the square root of the attention dimension.

The resulting values are passed through a softmax function to obtain attention weights.

The attention weights are then averaged to obtain a relative attention score for each word.

### 7. Visualization

The calculated attention scores are normalized and displayed as progress bars.

The word with the highest calculated attention score is displayed separately at the end of the application.

## Attention Formula

The project demonstrates the basic scaled dot-product attention concept:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

Where:

* Q represents Query
* K represents Key
* V represents Value
* `d_k` represents the dimension of the Key vectors

The project calculates the attention weights and uses them to obtain a relative score for each word.

## File Description

### app.py

Contains the Streamlit user interface and controls the complete application workflow.

It handles:

* Image upload
* Image display
* OCR processing
* Word extraction
* Embedding generation
* Attention calculation
* Attention visualization
* Highest calculated attention display

### ocr.py

Handles text extraction from the uploaded image using Tesseract OCR.

### embedding.py

Loads the `all-MiniLM-L6-v2` Sentence Transformer model and generates embeddings for the extracted words.

### attention.py

Implements the scaled dot-product attention calculation using Query, Key, and Value projections.

### requirements.txt

Contains the Python packages required to run the project.

## Usage

1. Start the Streamlit application.
2. Open the local Streamlit URL.
3. Upload a study-notes image.
4. Wait for OCR processing.
5. View the extracted text.
6. The application generates embeddings for the extracted words.
7. The attention mechanism calculates relative attention scores.
8. View the attention scores using the progress bars.
9. Check the word with the highest calculated attention.

## Example Input

A suitable input image can contain:

```text
Artificial Intelligence is transforming education.
Machine learning helps computers learn from data.
Deep learning uses neural networks to solve complex problems.
```

The application extracts these words and generates attention scores for the processed words.

## Example Output

The application displays:

```text
Extracted Text

Artificial Intelligence is transforming education.
Machine learning helps computers learn from data.

Word Attention

Artificial       [##########]
Intelligence     [###############]
transforming     [########]
education        [###########]
Machine          [#######]
learning         [############]

Highest Calculated Attention

Intelligence
```

The actual highest-attention word depends on the calculated attention values.

## Important Technical Note

This project uses `all-MiniLM-L6-v2` to generate embeddings for individual words.

`all-MiniLM-L6-v2` is primarily designed as a sentence embedding model. Using it with individual words in this project is a simplified approach intended to demonstrate the relationship between embeddings and an attention mechanism.

The attention mechanism implemented in this project is also a simplified educational implementation.

The Query, Key, and Value projection matrices are randomly initialized rather than learned from training data.

Therefore, the highest calculated attention word should not be interpreted as a reliable measure of semantic importance.

The project demonstrates the mechanics and visualization of attention rather than reproducing the internal attention weights of a pretrained Transformer model.

## Limitations

* OCR accuracy depends on the quality and clarity of the input image.
* Handwritten notes may not be recognized accurately.
* Only the first 20 suitable words are processed.
* The embedding model is primarily intended for sentence embeddings.
* The attention projection matrices are randomly initialized.
* The calculated attention score is not a learned semantic importance score.
* The project does not display the internal attention weights of a pretrained Transformer model.
* Tesseract OCR must be installed separately.

## Future Enhancements

Possible improvements include:

* Use actual Transformer token-level attention
* Highlight attention directly on extracted text
* Display OCR bounding boxes
* Support handwritten notes more effectively
* Add Tamil and other language OCR support
* Add sentence-level attention visualization
* Add keyword extraction
* Add topic classification
* Add text summarization
* Add question answering from study notes
* Allow users to compare attention across different images
* Add downloadable analysis reports
* Improve OCR preprocessing for low-quality images

## Learning Objectives

This project helps demonstrate the following concepts:

* Optical Character Recognition
* Text preprocessing
* Word embeddings
* Sentence Transformers
* Vector representations
* Query, Key, and Value
* Scaled dot-product attention
* Softmax
* Attention weights
* Streamlit application development
* Basic AI pipeline design

## AI Pipeline

The complete pipeline can be summarized as:

```text
Image
  |
  v
OCR
  |
  v
Text
  |
  v
Words
  |
  v
384-D Embeddings
  |
  v
Query / Key / Value
  |
  v
Scaled Dot-Product Attention
  |
  v
Attention Scores
  |
  v
Visualization
```

## Conclusion

AI Attention Visualizer provides a simple visual demonstration of how information can move from an image through OCR and embeddings into an attention mechanism.

The project is designed as an educational implementation to help understand the basic concepts behind attention mechanisms and how numerical representations of text can be processed and visualized.

