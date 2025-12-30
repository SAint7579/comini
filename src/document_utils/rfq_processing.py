"""
RFQ (Request for Quote) Processing Module

Handles parsing of RFQ files (CSV, Excel, PDF) and converts them
into structured search queries using an LLM.

Supports chunked processing for large files with async parallelization.
"""

import sys
import tempfile
from pathlib import Path
import asyncio
from typing import BinaryIO

import dotenv
dotenv.load_dotenv()

# Add src to path for direct script execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from std_utils import get_logger

logger = get_logger("comini.rfq")


# ============================================================================
# Configuration
# ============================================================================

# Maximum rows per chunk for LLM processing (for CSV/Excel)
CHUNK_SIZE = 50  # Adjust based on token limits and row complexity

# Maximum characters per chunk for PDF text processing
PDF_CHUNK_SIZE = 4000  # ~1000 tokens, safe for context window

MAX_CONCURRENT_CHUNKS = 5  # Limit concurrent API calls


# ============================================================================
# Pydantic Models
# ============================================================================

class RFQItem(BaseModel):
    """A single item extracted from an RFQ document."""
    
    raw_text: str = Field(
        description="The original raw text from the RFQ for this item"
    )
    search_query: str = Field(
        description="A cleaned, expanded search query optimized for product matching. "
                    "Expand abbreviations and normalize product terminology."
    )
    quantity: int | None = Field(
        default=None,
        description="The requested quantity if specified"
    )
    unit: str | None = Field(
        default=None,
        description="The unit of measurement (e.g., Stück, pcs, kg)"
    )
    notes: str | None = Field(
        default=None,
        description="Any additional notes or specifications"
    )


class ChunkResult(BaseModel):
    """Result from processing a single chunk."""
    
    items: list[RFQItem] = Field(
        description="List of items extracted from this chunk"
    )


class RFQProcessingResult(BaseModel):
    """Result of processing an RFQ document."""
    
    items: list[RFQItem] = Field(
        description="List of items extracted from the RFQ"
    )
    source_type: str = Field(
        description="Type of source file (csv, xlsx, pdf)"
    )
    total_items: int = Field(
        description="Total number of items extracted"
    )
    chunks_processed: int = Field(
        default=1,
        description="Number of chunks the document was split into"
    )
    raw_content_preview: str = Field(
        description="Preview of the raw content (first 500 chars)"
    )


# ============================================================================
# LLM Configuration
# ============================================================================

RFQ_EXTRACTION_PROMPT = """You are an expert at parsing Request for Quote (RFQ) documents for industrial tools and fasteners.

## Your Task
Extract each line item from the RFQ data and convert it into a structured search query.

## Context: Industrial Tools & Fasteners
The catalog contains products from Würth Industrie including:
- Screws, bolts, nuts, washers (various types and materials)
- Screwdriver bits (Torx TX, Phillips PH, Pozidriv PZ, Slotted SL)
- Hand tools, power tools, measuring instruments
- Fastening systems, anchors, rivets

## Common Abbreviations to Expand
- TX → Torx, PH → Phillips, PZ → Pozidriv, SL → Slotted
- gv zn / galv. verzinkt → galvanisch verzinkt (galvanized)
- A2 / A4 → Edelstahl (stainless steel)
- M4, M5, M6 → Metric thread sizes
- ST → Stahl (steel), MS → Messing (brass)
- SKS → Senkschraube, ZYL → Zylinderschraube
- MU → Mutter (nut), SHR → Scheibe (washer)
- SORT → Sortiment (assortment), TLG → Teilig (pieces)
- 6KT → Sechskant (hexagon)

## Instructions
1. Parse each row/line as a separate item
2. Extract the raw text exactly as it appears
3. Create an optimized search query by:
   - Expanding abbreviations
   - Normalizing product names
   - Keeping important specifications (size, material, coating)
4. Extract quantity and unit if present
5. Note any special requirements or specifications

Return ALL items found in this chunk."""


def get_rfq_processor() -> ChatOpenAI:
    """Get the LLM model configured for RFQ processing."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )


# ============================================================================
# File Loading Functions
# ============================================================================

def load_csv(file_path: str | Path | BinaryIO) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(file_path)


def load_excel(file_path: str | Path | BinaryIO) -> pd.DataFrame:
    """Load an Excel file into a DataFrame."""
    return pd.read_excel(file_path)


def load_pdf(file_path: str | Path | BinaryIO) -> str:
    """
    Load a PDF file and extract text using pypdf.
    
    Args:
        file_path: Path to PDF file or file-like object
        
    Returns:
        Extracted text from PDF
    """
    from pypdf import PdfReader
    
    logger.info("Extracting text from PDF with pypdf")
    
    reader = PdfReader(file_path)
    
    text_parts = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
    
    text = "\n\n".join(text_parts)
    logger.info(f"PDF extracted: {len(text)} characters from {len(reader.pages)} pages")
    
    return text


# Commented out - marker requires large ML models
# def load_pdf_with_marker(file_path: str | Path | BinaryIO) -> str:
#     """
#     Load a PDF file and convert to markdown text using marker.
#     
#     Args:
#         file_path: Path to PDF file or file-like object
#         
#     Returns:
#         Extracted text in markdown format
#     """
#     from marker.converters.pdf import PdfConverter
#     from marker.models import create_model_dict
#     from marker.output import text_from_rendered
#     
#     # Handle file-like objects by writing to temp file
#     if hasattr(file_path, 'read'):
#         with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
#             tmp.write(file_path.read())
#             tmp_path = tmp.name
#         try:
#             return _convert_pdf_with_marker(tmp_path)
#         finally:
#             Path(tmp_path).unlink(missing_ok=True)
#     else:
#         return _convert_pdf_with_marker(str(file_path))
#
#
# def _convert_pdf_with_marker(pdf_path: str) -> str:
#     """Internal function to convert PDF using marker."""
#     from marker.converters.pdf import PdfConverter
#     from marker.models import create_model_dict
#     from marker.output import text_from_rendered
#     
#     logger.info(f"Converting PDF with marker: {pdf_path}")
#     
#     # Create model dict and converter
#     model_dict = create_model_dict()
#     converter = PdfConverter(artifact_dict=model_dict)
#     
#     # Convert PDF to markdown
#     rendered = converter(pdf_path)
#     text, _, _ = text_from_rendered(rendered)
#     
#     logger.info(f"PDF converted: {len(text)} characters extracted")
#     return text


def chunk_text(text: str, chunk_size: int = PDF_CHUNK_SIZE) -> list[str]:
    """
    Split text into chunks for LLM processing.
    
    Tries to split on paragraph boundaries for better context.
    
    Args:
        text: Text to split
        chunk_size: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    
    # Split by double newlines (paragraphs) first
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    for para in paragraphs:
        # If adding this paragraph exceeds limit, save current and start new
        if len(current_chunk) + len(para) + 2 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If single paragraph is too long, split by lines
            if len(para) > chunk_size:
                lines = para.split('\n')
                current_chunk = ""
                for line in lines:
                    if len(current_chunk) + len(line) + 1 > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = line
                    else:
                        current_chunk += '\n' + line if current_chunk else line
            else:
                current_chunk = para
        else:
            current_chunk += '\n\n' + para if current_chunk else para
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    logger.info(f"Split text into {len(chunks)} chunks of ~{chunk_size} chars")
    return chunks


def dataframe_to_text(df: pd.DataFrame, include_row_numbers: bool = True) -> str:
    """
    Convert a DataFrame to a text representation suitable for LLM processing.
    
    Args:
        df: DataFrame to convert
        include_row_numbers: Whether to include row numbers
    
    Returns:
        Text representation of the DataFrame
    """
    # Get column headers
    headers = " | ".join(str(col) for col in df.columns)
    
    # Convert each row to text
    rows = []
    for idx, row in df.iterrows():
        row_text = " | ".join(str(val) if pd.notna(val) else "" for val in row)
        if include_row_numbers:
            rows.append(f"Row {idx + 1}: {row_text}")
        else:
            rows.append(row_text)
    
    # Combine
    text = f"Columns: {headers}\n\n"
    text += "\n".join(rows)
    
    return text


def chunk_dataframe(df: pd.DataFrame, chunk_size: int = CHUNK_SIZE) -> list[pd.DataFrame]:
    """
    Split a DataFrame into smaller chunks.
    
    Args:
        df: DataFrame to split
        chunk_size: Maximum rows per chunk
        
    Returns:
        List of DataFrame chunks
    """
    if len(df) <= chunk_size:
        return [df]
    
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunks.append(chunk)
    
    logger.info(f"Split {len(df)} rows into {len(chunks)} chunks of ~{chunk_size} rows")
    return chunks


# ============================================================================
# Chunk Processing Functions
# ============================================================================

def process_single_chunk(
    chunk_df: pd.DataFrame,
    chunk_index: int,
    column_headers: str
) -> list[RFQItem]:
    """
    Process a single chunk of RFQ data.
    
    Args:
        chunk_df: DataFrame chunk to process
        chunk_index: Index of this chunk (for logging)
        column_headers: Column headers string (same for all chunks)
        
    Returns:
        List of RFQItems extracted from this chunk
    """
    logger.debug(f"Processing chunk {chunk_index + 1} with {len(chunk_df)} rows")
    
    # Convert chunk to text
    content_text = dataframe_to_text(chunk_df)
    
    # Set up LLM with structured output
    llm = get_rfq_processor()
    structured_llm = llm.with_structured_output(ChunkResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", RFQ_EXTRACTION_PROMPT),
        ("human", """Process this chunk of RFQ data and extract all items:

{content}

Extract each item with its raw text and optimized search query.""")
    ])
    
    chain = prompt | structured_llm
    
    # Process with LLM
    result = chain.invoke({"content": content_text})
    
    logger.debug(f"Chunk {chunk_index + 1}: extracted {len(result.items)} items")
    return result.items


async def process_chunk_async(
    chunk_df: pd.DataFrame,
    chunk_index: int,
    column_headers: str,
    semaphore: asyncio.Semaphore
) -> list[RFQItem]:
    """
    Async wrapper for processing a single chunk with concurrency control.
    """
    async with semaphore:
        logger.info(f"Starting chunk {chunk_index + 1}")
        items = await asyncio.to_thread(
            process_single_chunk,
            chunk_df,
            chunk_index,
            column_headers
        )
        logger.info(f"Completed chunk {chunk_index + 1}: {len(items)} items")
        return items


# ============================================================================
# PDF Text Chunk Processing Functions
# ============================================================================

def process_single_text_chunk(
    text_chunk: str,
    chunk_index: int
) -> list[RFQItem]:
    """
    Process a single text chunk from a PDF.
    
    Args:
        text_chunk: Text chunk to process
        chunk_index: Index of this chunk (for logging)
        
    Returns:
        List of RFQItems extracted from this chunk
    """
    logger.debug(f"Processing text chunk {chunk_index + 1} ({len(text_chunk)} chars)")
    
    # Set up LLM with structured output
    llm = get_rfq_processor()
    structured_llm = llm.with_structured_output(ChunkResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", RFQ_EXTRACTION_PROMPT),
        ("human", """Process this section of an RFQ document and extract all product requests.

This is text extracted from a PDF document. Each line or item may represent a product request.

Content:
{content}

Extract each item with its raw text and optimized search query. 
If no product requests are found in this section, return an empty list.""")
    ])
    
    chain = prompt | structured_llm
    
    # Process with LLM
    result = chain.invoke({"content": text_chunk})
    
    logger.debug(f"Text chunk {chunk_index + 1}: extracted {len(result.items)} items")
    return result.items


async def process_text_chunk_async(
    text_chunk: str,
    chunk_index: int,
    semaphore: asyncio.Semaphore
) -> list[RFQItem]:
    """
    Async wrapper for processing a single text chunk with concurrency control.
    """
    async with semaphore:
        logger.info(f"Starting text chunk {chunk_index + 1}")
        items = await asyncio.to_thread(
            process_single_text_chunk,
            text_chunk,
            chunk_index
        )
        logger.info(f"Completed text chunk {chunk_index + 1}: {len(items)} items")
        return items


async def process_pdf_text_async(
    text: str,
    chunk_size: int = PDF_CHUNK_SIZE,
    max_concurrent: int = MAX_CONCURRENT_CHUNKS
) -> RFQProcessingResult:
    """
    Process PDF text with chunking and async parallelization.
    
    Args:
        text: Full text extracted from PDF
        chunk_size: Maximum characters per chunk
        max_concurrent: Maximum concurrent LLM calls
        
    Returns:
        RFQProcessingResult with all extracted items
    """
    logger.info(f"Processing PDF text ({len(text)} chars, chunk_size={chunk_size})")
    
    # Split into chunks
    chunks = chunk_text(text, chunk_size)
    num_chunks = len(chunks)
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process all chunks in parallel
    logger.info(f"Processing {num_chunks} text chunks with max {max_concurrent} concurrent")
    
    tasks = [
        process_text_chunk_async(chunk, idx, semaphore)
        for idx, chunk in enumerate(chunks)
    ]
    
    # Gather results (maintains order)
    chunk_results = await asyncio.gather(*tasks)
    
    # Concatenate all items
    all_items: list[RFQItem] = []
    for items in chunk_results:
        all_items.extend(items)
    
    result = RFQProcessingResult(
        items=all_items,
        source_type="pdf",
        total_items=len(all_items),
        chunks_processed=num_chunks,
        raw_content_preview=text[:500]
    )
    
    logger.info(f"Extracted {result.total_items} items from {num_chunks} PDF chunks")
    
    return result


# ============================================================================
# Main Processing Functions
# ============================================================================

async def process_rfq_dataframe_async(
    df: pd.DataFrame,
    source_type: str = "dataframe",
    chunk_size: int = CHUNK_SIZE,
    max_concurrent: int = MAX_CONCURRENT_CHUNKS
) -> RFQProcessingResult:
    """
    Process an RFQ DataFrame with chunking and async parallelization.
    
    Args:
        df: DataFrame containing RFQ data
        source_type: Type of source file
        chunk_size: Maximum rows per chunk
        max_concurrent: Maximum concurrent LLM calls
        
    Returns:
        RFQProcessingResult with all extracted items
    """
    logger.info(f"Processing RFQ with {len(df)} rows (chunk_size={chunk_size})")
    
    # Get column headers (same for all chunks)
    column_headers = " | ".join(str(col) for col in df.columns)
    
    # Split into chunks
    chunks = chunk_dataframe(df, chunk_size)
    num_chunks = len(chunks)
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process all chunks in parallel
    logger.info(f"Processing {num_chunks} chunks with max {max_concurrent} concurrent")
    
    tasks = [
        process_chunk_async(chunk, idx, column_headers, semaphore)
        for idx, chunk in enumerate(chunks)
    ]
    
    # Gather results (maintains order)
    chunk_results = await asyncio.gather(*tasks)
    
    # Concatenate all items
    all_items: list[RFQItem] = []
    for items in chunk_results:
        all_items.extend(items)
    
    # Build full content preview
    full_content = dataframe_to_text(df)
    
    result = RFQProcessingResult(
        items=all_items,
        source_type=source_type,
        total_items=len(all_items),
        chunks_processed=num_chunks,
        raw_content_preview=full_content[:500]
    )
    
    logger.info(f"Extracted {result.total_items} items from {num_chunks} chunks")
    
    return result


def process_rfq_dataframe(
    df: pd.DataFrame,
    source_type: str = "dataframe",
    chunk_size: int = CHUNK_SIZE
) -> RFQProcessingResult:
    """
    Synchronous wrapper for process_rfq_dataframe_async.
    
    For small DataFrames (≤ chunk_size), processes directly.
    For larger ones, uses async chunked processing.
    """
    return asyncio.run(process_rfq_dataframe_async(df, source_type, chunk_size))


async def process_rfq_file_async(
    file_path: str | Path | BinaryIO,
    file_type: str | None = None,
    chunk_size: int | None = None
) -> RFQProcessingResult:
    """
    Process an RFQ file (CSV, Excel, or PDF) with async chunked processing.
    
    Args:
        file_path: Path to the file or file-like object
        file_type: File type ('csv', 'xlsx', 'xls', 'pdf'). Auto-detected if not provided.
        chunk_size: Maximum rows per chunk (CSV/Excel) or chars per chunk (PDF).
                    If None, uses defaults: CHUNK_SIZE for tabular, PDF_CHUNK_SIZE for PDF.
        
    Returns:
        RFQProcessingResult with extracted items
    """
    # Determine file type
    if file_type is None:
        if isinstance(file_path, (str, Path)):
            suffix = Path(file_path).suffix.lower()
            file_type = suffix.lstrip('.')
        else:
            raise ValueError("file_type must be specified for file-like objects")
    
    file_type = file_type.lower()
    
    # Handle PDF files separately
    if file_type == 'pdf':
        pdf_chunk_size = chunk_size or PDF_CHUNK_SIZE
        logger.info("Loading PDF file with marker")
        text = await asyncio.to_thread(load_pdf, file_path)
        return await process_pdf_text_async(text, chunk_size=pdf_chunk_size)
    
    # Handle tabular files (CSV, Excel)
    tabular_chunk_size = chunk_size or CHUNK_SIZE
    
    if file_type == 'csv':
        logger.info("Loading CSV file")
        df = await asyncio.to_thread(load_csv, file_path)
    elif file_type in ('xlsx', 'xls'):
        logger.info("Loading Excel file")
        df = await asyncio.to_thread(load_excel, file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Supported: csv, xlsx, xls, pdf")
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return await process_rfq_dataframe_async(df, source_type=file_type, chunk_size=tabular_chunk_size)


def process_rfq_file(
    file_path: str | Path | BinaryIO,
    file_type: str | None = None,
    chunk_size: int = CHUNK_SIZE
) -> RFQProcessingResult:
    """
    Synchronous wrapper for process_rfq_file_async.
    """
    return asyncio.run(process_rfq_file_async(file_path, file_type, chunk_size))


# ============================================================================
# CLI / Testing
# ============================================================================

if __name__ == "__main__":
    
    # Example usage
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        result = process_rfq_file(file_path)
        
        print(f"\n{'='*60}")
        print(f"RFQ Processing Result")
        print(f"{'='*60}")
        print(f"Source: {result.source_type}")
        print(f"Total items: {result.total_items}")
        print(f"Chunks processed: {result.chunks_processed}")
        print(f"\nItems:")
        
        for i, item in enumerate(result.items, 1):
            print(f"\n--- Item {i} ---")
            print(f"Raw: {item.raw_text}")
            print(f"Query: {item.search_query}")
            if item.quantity:
                print(f"Qty: {item.quantity} {item.unit or ''}")
            if item.notes:
                print(f"Notes: {item.notes}")
    else:
        # Demo with sample data (larger to show chunking)
        sample_data = pd.DataFrame({
            'Position': list(range(1, 11)),
            'Beschreibung': [
                'TX25 Bit gv zn',
                'M6x20 SKS A2',
                'PH2 Schraubendreher 150mm',
                'M8 6KT Mutter verzinkt',
                'SORT 14-TLG Bits TX',
                'M4x10 ZYL INB A4',
                'PZ1 Bit 25mm',
                'SL 5.5 Schlitz Bit',
                'M10 Scheibe DIN 125 ST',
                'HX4 Innensechskant Schlüssel'
            ],
            'Menge': [100, 500, 10, 200, 5, 1000, 50, 50, 300, 2],
            'Einheit': ['Stück'] * 10
        })
        
        print("Processing sample RFQ data:")
        print(sample_data.to_string())
        print("\n")
        
        # Use small chunk size to demonstrate chunking
        result = process_rfq_dataframe(sample_data, source_type="demo", chunk_size=5)
        
        print(f"Extracted {result.total_items} items from {result.chunks_processed} chunks:\n")
        for i, item in enumerate(result.items, 1):
            print(f"{i}. Raw: {item.raw_text}")
            print(f"   Query: {item.search_query}")
            print()
