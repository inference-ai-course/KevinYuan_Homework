# Data Cleaning Pipeline Statistics

## Overview

- **Total Documents Loaded**: 408
- **Documents Processed**: 407
- **Documents Removed**: 1

## Token Count

- **Total Tokens Before**: 2,166,540
- **Total Tokens After**: 1,669,828
- **Tokens Removed**: 496,712
- **Removal Percentage**: 22.93%

## Cleaning Steps

1. **Language Detection**: Removed 1 non-English documents
2. **HTML Noise Removal**: Cleaned 404 documents
3. **Deduplication (MinHash)**: Removed 0 duplicate documents
4. **PII Removal**: Removed 54287 PII instances
5. **Repetitive N-grams**: Removed 0 documents with excessive repetition

## Processing Pipeline

```
Input Data → Language Detection → HTML Stripping → MinHash Deduplication → PII Removal → N-gram Filtering → Clean Corpus
```

## Configuration

- **Language**: English only
- **Deduplication Threshold**: 0.7 (MinHash similarity)
- **N-gram Size**: 5
- **Max Repetition Ratio**: 0.3
