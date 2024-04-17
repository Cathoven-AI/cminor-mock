# CMinor: Text Analyzer Mock

CMinor is a Python package developed by Cathoven AI designed to analyze texts. It provides functionalities such as determining CEFR (Common European Framework of Reference for Languages) levels, generating questions based on texts, and more. Please note that this package serves as a mock version of the actual CMinor system developed by Cathoven AI.

## Features

- **CEFR Level Detection**: Determine the CEFR level of a given text.
- **Question Generation**: Generate questions based on the content of a text.
- **Mock Functionality**: This package mimics some features of the actual CMinor system developed by Cathoven AI.
- **Easy to Use**: Simple and straightforward APIs for text analysis tasks.

## Installation

You can install CMinor via pip:

```bash
pip install cminor
```
# Usage
``` python
# Analyze CEFR level of the given text with high accuracy

from cminor.analyzer import AdoTextAnalyzer

analyzer = AdoTextAnalyzer()
text = "Example long text"

result = analyzer.analyze_cefr(text) #JSON formatted result

```
# Contact
We are serving APIs for enterprises. Please contact us at [contact@cathoven.com](mailto:contact@cathoven.com)

# Important Note
This is a non-complete mock version of cminor. Your text will not be analyzed here. Please visit [Cathoven A.I](https://hub.cathoven.com) to give it a try.