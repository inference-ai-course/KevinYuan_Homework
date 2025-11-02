"""
Data Cleaning Pipeline
Processes JSON, TXT, and JSONL files through multiple cleaning stages:
1. Language Detection
2. HTML Noise Removal
3. MinHash Deduplication (similarity >= 0.7)
4. PII Removal
5. Repetitive N-gram Removal
"""

import json
import re
import os
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import Counter
import logging

# Core libraries
from langdetect import detect, LangDetectException
from datasketch import MinHash, MinHashLSH

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCleaningPipeline:
    def __init__(self, input_dir: str, output_dir: str = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'total_documents': 0,
            'total_tokens_before': 0,
            'total_tokens_after': 0,
            'removed_non_english': 0,
            'removed_duplicates': 0,
            'removed_pii_instances': 0,
            'removed_repetitive_ngrams': 0,
            'documents_with_html': 0,
            'documents_processed': 0
        }
        
    def load_documents(self) -> List[Dict[str, str]]:
        """Load all documents from JSON, TXT, and JSONL files."""
        documents = []
        
        # Load JSON file (arxiv_clean_formatted.json)
        json_file = self.input_dir / 'arxiv_clean_formatted.json'
        if json_file.exists():
            logger.info(f"Loading JSON file: {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    text = f"{item.get('title', '')} {item.get('abstract', '')}"
                    documents.append({
                        'text': text,
                        'source': str(json_file),
                        'type': 'json'
                    })
        
        # Load TXT files from pdf_ocr directory
        pdf_ocr_dir = self.input_dir / 'pdf_ocr'
        if pdf_ocr_dir.exists():
            logger.info(f"Loading TXT files from: {pdf_ocr_dir}")
            for txt_file in pdf_ocr_dir.glob('*.txt'):
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    documents.append({
                        'text': text,
                        'source': str(txt_file),
                        'type': 'txt'
                    })
        
        # Load JSONL files from transcription directory
        transcription_dir = self.input_dir / 'transcription'
        if transcription_dir.exists():
            logger.info(f"Loading JSONL files from: {transcription_dir}")
            for jsonl_file in transcription_dir.glob('*.jsonl'):
                texts = []
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if 'text' in data:
                                texts.append(data['text'])
                        except json.JSONDecodeError:
                            continue
                if texts:
                    documents.append({
                        'text': ' '.join(texts),
                        'source': str(jsonl_file),
                        'type': 'jsonl'
                    })
        
        self.stats['total_documents'] = len(documents)
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def detect_language(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Filter documents to keep only English text."""
        logger.info("Step 1: Detecting language...")
        english_docs = []
        
        for doc in documents:
            text = doc['text'].strip()
            if not text or len(text) < 10:
                self.stats['removed_non_english'] += 1
                continue
                
            try:
                lang = detect(text)
                if lang == 'en':
                    english_docs.append(doc)
                else:
                    self.stats['removed_non_english'] += 1
            except LangDetectException:
                # If detection fails, keep the document
                english_docs.append(doc)
        
        logger.info(f"Kept {len(english_docs)} English documents, removed {self.stats['removed_non_english']} non-English")
        return english_docs
    
    def strip_html_noise(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove HTML tags and other noise."""
        logger.info("Step 2: Stripping HTML noise...")
        
        html_pattern = re.compile(r'<[^>]+>')
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        
        for doc in documents:
            original_text = doc['text']
            
            # Remove HTML tags
            text = html_pattern.sub(' ', original_text)
            
            # Remove URLs
            text = url_pattern.sub(' ', text)
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep sentence structure
            text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\'\"]', ' ', text)
            
            # Clean up again
            text = re.sub(r'\s+', ' ', text).strip()
            
            if original_text != text:
                self.stats['documents_with_html'] += 1
            
            doc['text'] = text
        
        logger.info(f"HTML noise removed from {self.stats['documents_with_html']} documents")
        return documents
    
    def deduplicate_minhash(self, documents: List[Dict[str, str]], threshold: float = 0.7) -> List[Dict[str, str]]:
        """Remove near-duplicate documents using MinHash LSH."""
        logger.info(f"Step 3: Deduplicating with MinHash (threshold >= {threshold})...")
        
        # Create MinHash LSH index
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        minhashes = {}
        
        # Generate MinHash for each document
        for idx, doc in enumerate(documents):
            m = MinHash(num_perm=128)
            # Tokenize text into words
            tokens = doc['text'].lower().split()
            for token in tokens:
                m.update(token.encode('utf-8'))
            
            minhashes[idx] = m
            lsh.insert(idx, m)
        
        # Find duplicates
        seen = set()
        unique_docs = []
        
        for idx, doc in enumerate(documents):
            if idx in seen:
                continue
            
            # Query for similar documents
            result = lsh.query(minhashes[idx])
            
            # Mark all similar documents as seen (except the first one)
            for similar_idx in result:
                if similar_idx != idx:
                    seen.add(similar_idx)
            
            unique_docs.append(doc)
        
        self.stats['removed_duplicates'] = len(documents) - len(unique_docs)
        logger.info(f"Kept {len(unique_docs)} unique documents, removed {self.stats['removed_duplicates']} duplicates")
        return unique_docs
    
    def remove_pii(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove PII: emails, credit card numbers, phone numbers."""
        logger.info("Step 4: Removing PII...")
        
        # Regex patterns for PII
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Credit card pattern (basic)
        cc_pattern = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
        
        # Phone number patterns (various formats)
        phone_patterns = [
            re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),  # 123-456-7890
            re.compile(r'\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b'),  # (123) 456-7890
            re.compile(r'\b\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b')  # International
        ]
        
        pii_count = 0
        
        for doc in documents:
            text = doc['text']
            
            # Remove emails
            matches = len(email_pattern.findall(text))
            if matches > 0:
                pii_count += matches
                text = email_pattern.sub('[EMAIL]', text)
            
            # Remove credit cards
            matches = len(cc_pattern.findall(text))
            if matches > 0:
                pii_count += matches
                text = cc_pattern.sub('[CREDIT_CARD]', text)
            
            # Remove phone numbers
            for pattern in phone_patterns:
                matches = len(pattern.findall(text))
                if matches > 0:
                    pii_count += matches
                    text = pattern.sub('[PHONE]', text)
            
            doc['text'] = text
        
        self.stats['removed_pii_instances'] = pii_count
        logger.info(f"Removed {pii_count} PII instances")
        return documents
    
    def remove_repetitive_ngrams(self, documents: List[Dict[str, str]], n: int = 5, max_ratio: float = 0.3) -> List[Dict[str, str]]:
        """Remove documents with excessive repetitive n-grams."""
        logger.info(f"Step 5: Removing repetitive {n}-grams (max ratio: {max_ratio})...")
        
        clean_docs = []
        removed = 0
        
        for doc in documents:
            text = doc['text']
            words = text.split()
            
            if len(words) < n:
                clean_docs.append(doc)
                continue
            
            # Generate n-grams
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
            
            if not ngrams:
                clean_docs.append(doc)
                continue
            
            # Count n-gram frequencies
            ngram_counts = Counter(ngrams)
            
            # Find most common n-gram
            if ngram_counts:
                most_common_count = ngram_counts.most_common(1)[0][1]
                repetition_ratio = most_common_count / len(ngrams)
                
                if repetition_ratio <= max_ratio:
                    clean_docs.append(doc)
                else:
                    removed += 1
        
        self.stats['removed_repetitive_ngrams'] = removed
        logger.info(f"Kept {len(clean_docs)} documents, removed {removed} with excessive repetition")
        return clean_docs
    
    def count_tokens(self, text: str) -> int:
        """Simple token counter (word-based)."""
        return len(text.split())
    
    def save_clean_corpus(self, documents: List[Dict[str, str]]) -> None:
        """Save cleaned corpus to a single text file."""
        output_file = self.output_dir / 'clean_corpus.txt'
        
        logger.info(f"Saving clean corpus to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                text = doc['text']
                self.stats['total_tokens_after'] += self.count_tokens(text)
                f.write(text + '\n\n')
        
        self.stats['documents_processed'] = len(documents)
        logger.info(f"Saved {len(documents)} documents to {output_file}")
    
    def save_statistics(self) -> None:
        """Save statistics to markdown file."""
        stats_file = self.output_dir / 'stats.md'
        
        logger.info(f"Saving statistics to {stats_file}")
        
        removal_percentage = 0
        if self.stats['total_tokens_before'] > 0:
            removal_percentage = ((self.stats['total_tokens_before'] - self.stats['total_tokens_after']) 
                                / self.stats['total_tokens_before'] * 100)
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("# Data Cleaning Pipeline Statistics\n\n")
            f.write("## Overview\n\n")
            f.write(f"- **Total Documents Loaded**: {self.stats['total_documents']}\n")
            f.write(f"- **Documents Processed**: {self.stats['documents_processed']}\n")
            f.write(f"- **Documents Removed**: {self.stats['total_documents'] - self.stats['documents_processed']}\n\n")
            
            f.write("## Token Count\n\n")
            f.write(f"- **Total Tokens Before**: {self.stats['total_tokens_before']:,}\n")
            f.write(f"- **Total Tokens After**: {self.stats['total_tokens_after']:,}\n")
            f.write(f"- **Tokens Removed**: {self.stats['total_tokens_before'] - self.stats['total_tokens_after']:,}\n")
            f.write(f"- **Removal Percentage**: {removal_percentage:.2f}%\n\n")
            
            f.write("## Cleaning Steps\n\n")
            f.write(f"1. **Language Detection**: Removed {self.stats['removed_non_english']} non-English documents\n")
            f.write(f"2. **HTML Noise Removal**: Cleaned {self.stats['documents_with_html']} documents\n")
            f.write(f"3. **Deduplication (MinHash)**: Removed {self.stats['removed_duplicates']} duplicate documents\n")
            f.write(f"4. **PII Removal**: Removed {self.stats['removed_pii_instances']} PII instances\n")
            f.write(f"5. **Repetitive N-grams**: Removed {self.stats['removed_repetitive_ngrams']} documents with excessive repetition\n\n")
            
            f.write("## Processing Pipeline\n\n")
            f.write("```\n")
            f.write("Input Data → Language Detection → HTML Stripping → MinHash Deduplication → PII Removal → N-gram Filtering → Clean Corpus\n")
            f.write("```\n\n")
            
            f.write("## Configuration\n\n")
            f.write("- **Language**: English only\n")
            f.write("- **Deduplication Threshold**: 0.7 (MinHash similarity)\n")
            f.write("- **N-gram Size**: 5\n")
            f.write("- **Max Repetition Ratio**: 0.3\n")
        
        logger.info("Statistics saved")
    
    def run(self) -> None:
        """Execute the complete data cleaning pipeline."""
        logger.info("=" * 80)
        logger.info("Starting Data Cleaning Pipeline")
        logger.info("=" * 80)
        
        # Load documents
        documents = self.load_documents()
        
        # Calculate initial token count
        for doc in documents:
            self.stats['total_tokens_before'] += self.count_tokens(doc['text'])
        
        logger.info(f"Initial token count: {self.stats['total_tokens_before']:,}")
        
        # Run cleaning steps
        documents = self.detect_language(documents)
        documents = self.strip_html_noise(documents)
        documents = self.deduplicate_minhash(documents, threshold=0.7)
        documents = self.remove_pii(documents)
        documents = self.remove_repetitive_ngrams(documents, n=5, max_ratio=0.3)
        
        # Save results
        self.save_clean_corpus(documents)
        self.save_statistics()
        
        logger.info("=" * 80)
        logger.info("Pipeline Complete!")
        logger.info(f"Clean corpus saved to: {self.output_dir / 'clean_corpus.txt'}")
        logger.info(f"Statistics saved to: {self.output_dir / 'stats.md'}")
        logger.info("=" * 80)


if __name__ == "__main__":
    # Set input directory
    input_dir = r"c:\Users\Kevin\Desktop\ai_class\KevinYuan_Homework\week2_HW\4_dd"
    
    # Create and run pipeline
    pipeline = DataCleaningPipeline(input_dir)
    pipeline.run()
