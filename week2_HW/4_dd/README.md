# Data Cleaning Pipeline - Results

## Overview
This folder contains the results of a comprehensive data cleaning pipeline applied to three types of data sources:
- **JSON**: `arxiv_clean_formatted.json` (400 documents)
- **TXT**: PDF OCR results from `pdf_ocr/` directory (numerous text files)
- **JSONL**: Transcription data from `transcription/` directory (8 files)

## Pipeline Steps

The data cleaning pipeline (`data_cleaning_pipeline.py`) processes all documents through 5 sequential steps:

### 1. Language Detection (using `langdetect`)
- Filters out non-English documents
- Ensures only English content is processed
- **Result**: Removed 1 non-English document

### 2. HTML Noise Removal
- Strips HTML tags using regex patterns
- Removes URLs and web links
- Cleans excessive whitespace
- Normalizes special characters while preserving sentence structure
- **Result**: Cleaned 404 documents

### 3. MinHash Deduplication (using `datasketch`)
- Implements MinHash LSH (Locality-Sensitive Hashing)
- Identifies near-duplicate documents with similarity >= 0.7
- Uses 128 permutations for accurate similarity detection
- **Result**: No duplicates found (all documents were unique)

### 4. PII Removal
- Removes email addresses
- Removes credit card numbers
- Removes phone numbers (multiple formats)
- Replaces with placeholders: [EMAIL], [CREDIT_CARD], [PHONE]
- **Result**: Removed 54,287 PII instances

### 5. Repetitive N-gram Filtering
- Detects excessive repetition of 5-grams
- Removes documents with repetition ratio > 0.3
- Ensures content quality and diversity
- **Result**: No documents removed (all passed quality check)

## Output Files

### 1. `clean_corpus.txt`
The final cleaned corpus containing all processed documents. Each document is separated by double newlines.

**Key Statistics:**
- 407 documents processed
- 1,669,828 tokens (cleaned)
- High-quality, de-duplicated, PII-free text

### 2. `stats.md`
Detailed statistics report containing:
- Document counts (loaded, processed, removed)
- Token counts (before/after cleaning)
- Removal percentages
- Step-by-step cleaning metrics
- Configuration parameters

## Key Results

| Metric | Value |
|--------|-------|
| Total Documents Loaded | 408 |
| Documents Processed | 407 |
| Total Tokens Before | 2,166,540 |
| Total Tokens After | 1,669,828 |
| Tokens Removed | 496,712 |
| **Removal Percentage** | **22.93%** |

## Dependencies

```bash
pip install langdetect datasketch
```

## Usage

To run the pipeline:

```bash
python data_cleaning_pipeline.py
```

## Configuration

The pipeline uses the following default parameters:
- **Language**: English only
- **Deduplication Threshold**: 0.7 (MinHash similarity)
- **N-gram Size**: 5
- **Max Repetition Ratio**: 0.3

These can be modified in the `run()` method of the `DataCleaningPipeline` class.

## Technical Details

### Core Libraries
- **langdetect**: Robust language detection based on Google's language-detection library
- **datasketch**: MinHash implementation for efficient near-duplicate detection

### Performance
- Total processing time: ~20 seconds
- Handles 408 documents with 2M+ tokens efficiently
- Scalable architecture for larger datasets

## Notes

- The pipeline is fully automated and requires no manual intervention
- All text is processed in UTF-8 encoding with error handling
- The system preserves document source information for traceability
- PII removal uses comprehensive regex patterns for multiple formats
- MinHash LSH provides probabilistic near-duplicate detection with high accuracy
