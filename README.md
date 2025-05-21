**Language Classifier**

This project implements a multi-language text classification system capable of identifying the
language of input text with high accuracy. The system employs a dual-feature extraction approach,
combining word-level and character-level TF-IDF vectorization to capture both semantic and
morphological patterns unique to different languages.

The project utilizes a Linear Support Vector Machine (LinearSVC) classifier trained on a
diverse corpus of multilingual text. By leveraging both word and character n-grams,
it achieves robust language detection even for short text fragments or those
containing specialized vocabulary. Text preprocessing includes lowercase conversion and
punctuation removal to normalize inputs across languages.

*Contributors*

Krish Mahajan

Aryan Ranjan
