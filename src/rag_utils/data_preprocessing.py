"""
Product data preprocessing with LLM-based feature extraction.

This module uses an LLM to extract structured product features from raw product data,
including standards, materials, dimensions, categories, brands, and applications.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import BinaryIO

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tqdm import tqdm

from std_utils import get_logger

logger = get_logger("comini.preprocessing")

# LLM Configuration
EXTRACTION_MODEL = "gpt-5.1"


# ============================================================================
# Pydantic Models for Structured Product Features
# ============================================================================

class ProductFeatures(BaseModel):
    """Structured product features extracted by LLM."""
    
    standard: str | None = Field(
        default=None,
        description="Industry standards and norms (e.g., DIN 3121, ISO 2725, VDE). Leave null if none mentioned."
    )
    
    materials: str | None = Field(
        default=None,
        description="Plain text description of materials used (e.g., 'Chrome Molybdenum steel with phosphate coating', 'Hardened carbon steel'). Include surface treatments and coatings."
    )
    
    size: str | None = Field(
        default=None,
        description="Size specifications (e.g., 'M6', '1/4 inch', 'TX25', 'SW13'). Include drive sizes, bit sizes, key sizes."
    )
    
    dimensions: str | None = Field(
        default=None,
        description="Physical dimensions (e.g., 'Length 25mm, Diameter 6.3mm'). Include length, width, height, diameter, weight."
    )
    
    category: str | None = Field(
        default=None,
        description="Product category and type (e.g., 'Screwdriver Bit', 'Socket Wrench', 'Hammer', 'Oil Filter Wrench')."
    )
    
    brand: str | None = Field(
        default=None,
        description="Brand or manufacturer name if mentioned (e.g., 'ZEBRA', 'Halder', 'Klingspor', 'Würth'). Leave null if not specified."
    )
    
    application: str | None = Field(
        default=None,
        description="Intended use and application (e.g., 'For TX screws in automotive assembly', 'For loosening oil filters', 'For impact drivers')."
    )
    
    additional_info: str | None = Field(
        default=None,
        description="Any other relevant information not covered above (special features, compatibility notes, included accessories, set contents)."
    )


class ProcessedProduct(BaseModel):
    """Complete processed product with all extracted features."""
    
    article_number: str = Field(description="Unique product article number")
    headline: str = Field(description="Product headline/title")
    long_description: str | None = Field(default=None, description="Detailed product description")
    image_url: str | None = Field(default=None, description="Product image URL")
    
    # LLM-extracted features
    standard: str | None = None
    materials: str | None = None
    size: str | None = None
    dimensions: str | None = None
    category: str | None = None
    brand: str | None = None
    application: str | None = None
    additional_info: str | None = None


# ============================================================================
# LLM Feature Extraction
# ============================================================================

EXTRACTION_PROMPT = """You are an expert in industrial tools, fasteners, and hardware. 
Your task is to analyze product information and extract structured features.

## Context
This is a product from Würth Industry, a supplier of industrial tools, fasteners, and assembly technology.

## Product Information
**Headline:** {headline}
**Short Description:** {short_description}
**Long Description:** {long_description}
**Technical Information:** {technical_info}
**Category Path:** {breadcrumbs}

## Instructions
Extract the following structured information from the product data:
1. **Standard**: Industry standards/norms (DIN, ISO, VDE, etc.)
2. **Materials**: Materials used, surface treatments, coatings (plain text description)
3. **Size**: Size specifications (drive size, bit size, key size, thread size)
4. **Dimensions**: Physical measurements (length, width, height, diameter, weight)
5. **Category**: Product type/category
6. **Brand**: Brand/manufacturer if mentioned
7. **Application**: Intended use and applications
8. **Additional Info**: Any other relevant details not in other categories

Be precise and extract only what is explicitly stated. Use null for fields with no relevant information."""


def get_extraction_llm() -> ChatOpenAI:
    """Get the LLM configured for feature extraction."""
    return ChatOpenAI(
        model=EXTRACTION_MODEL,
        temperature=0,
    )


def extract_product_features(
    headline: str,
    short_description: str,
    long_description: str,
    technical_info: str,
    breadcrumbs: str
) -> ProductFeatures:
    """
    Extract structured features from a product using LLM.
    
    Args:
        headline: Product headline/title
        short_description: Short product description
        long_description: Detailed product description
        technical_info: Technical specifications
        breadcrumbs: Category path
        
    Returns:
        ProductFeatures with extracted information
    """
    llm = get_extraction_llm()
    structured_llm = llm.with_structured_output(ProductFeatures)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACTION_PROMPT),
        ("human", "Please extract the product features.")
    ])
    
    # Chain the prompt with the structured LLM
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({
            "headline": headline or "",
            "short_description": short_description or "",
            "long_description": long_description or "",
            "technical_info": technical_info or "",
            "breadcrumbs": breadcrumbs or "",
        })
        return result
    except Exception as e:
        logger.error(f"LLM extraction error: {e}")
        # Return empty features on error
        return ProductFeatures()


def extract_product_features_batch(
    products: list[dict],
    max_workers: int = 5
) -> list[ProductFeatures]:
    """
    Extract features for multiple products in parallel.
    
    Args:
        products: List of product dicts with headline, short_description, etc.
        max_workers: Number of parallel LLM calls
        
    Returns:
        List of ProductFeatures in same order as input
    """
    import time
    
    total = len(products)
    features = [None] * total
    success_count = 0
    error_count = 0
    start_time = time.time()
    last_log_time = start_time
    
    # Log every N products or every M seconds
    log_interval_count = max(1, total // 20)  # Log ~20 times during processing
    log_interval_secs = 10  # Also log at least every 10 seconds
    
    logger.info(f"Starting LLM feature extraction for {total} products with {max_workers} workers")
    
    def extract_single(idx: int, product: dict) -> tuple[int, ProductFeatures, bool, str]:
        """Extract features for a single product. Returns (idx, features, success, headline)."""
        headline = product.get("headline", "")[:60]
        try:
            result = extract_product_features(
                headline=product.get("headline", ""),
                short_description=product.get("short_description", ""),
                long_description=product.get("long_description", ""),
                technical_info=product.get("technical_info", ""),
                breadcrumbs=product.get("breadcrumbs", ""),
            )
            # Check if we got meaningful data
            has_data = any([
                result.standard, result.materials, result.size,
                result.dimensions, result.category, result.brand,
                result.application, result.additional_info
            ])
            return idx, result, has_data, headline
        except Exception as e:
            logger.error(f"[{idx+1}/{total}] FAILED: {headline} - {e}")
            return idx, ProductFeatures(), False, headline
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_single, idx, product): idx
            for idx, product in enumerate(products)
        }
        
        for future in as_completed(futures):
            idx, result, success, headline = future.result()
            features[idx] = result
            
            if success:
                success_count += 1
            else:
                error_count += 1
            
            completed = success_count + error_count
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Log progress at intervals
            should_log = (
                completed % log_interval_count == 0 or
                current_time - last_log_time >= log_interval_secs or
                completed == total
            )
            
            if should_log:
                last_log_time = current_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate if rate > 0 else 0
                pct = completed / total * 100
                
                logger.info(
                    f"Progress: {completed}/{total} ({pct:.1f}%) | "
                    f"OK: {success_count} | ERR: {error_count} | "
                    f"Speed: {rate:.1f}/s | ETA: {eta:.0f}s"
                )
    
    total_time = time.time() - start_time
    logger.info(
        f"Feature extraction COMPLETE: {success_count} successful, {error_count} errors "
        f"out of {total} products in {total_time:.1f}s ({total/total_time:.1f} products/s)"
    )
    
    return features


# ============================================================================
# Image URL Extraction
# ============================================================================

def extract_image_url(page_url: str, timeout: int = 10) -> str | None:
    """
    Scrape the product page and extract the image URL from the socialshare img tag.
    
    Args:
        page_url: The product page URL
        timeout: Request timeout in seconds
        
    Returns:
        The image URL or None if not found
    """
    try:
        response = requests.get(page_url, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        img_tag = soup.find('img', class_='img-fluid js-socialshare-media')
        
        if img_tag and img_tag.get('src'):
            return img_tag['src']
        return None
    except Exception as e:
        logger.warning(f"Error fetching image from {page_url}: {e}")
        return None


def fetch_images_parallel(urls: list[str], max_workers: int = 10) -> list[str | None]:
    """
    Fetch image URLs from product pages in parallel.
    
    Args:
        urls: List of product page URLs
        max_workers: Number of parallel requests
        
    Returns:
        List of image URLs (or None) in same order as input
    """
    image_urls = [None] * len(urls)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(extract_image_url, url): idx 
            for idx, url in enumerate(urls) if pd.notna(url)
        }
        
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Fetching images"):
            idx = future_to_idx[future]
            image_urls[idx] = future.result()
    
    return image_urls


# ============================================================================
# Main Processing Function
# ============================================================================

def process_products(
    df: pd.DataFrame,
    max_workers_llm: int = 5,
    max_workers_images: int = 10,
    fetch_images: bool = True
) -> pd.DataFrame:
    """
    Process product data with LLM feature extraction.
    
    Args:
        df: Input dataframe with product data
        max_workers_llm: Number of parallel LLM extraction threads
        max_workers_images: Number of parallel image fetch threads
        fetch_images: Whether to fetch image URLs
        
    Returns:
        DataFrame with extracted features:
        - article_number, headline, long_description, image_url
        - standard, materials, size, dimensions, category, brand, application, additional_info
    """
    logger.info(f"Processing {len(df)} products with LLM feature extraction...")
    
    # Clean article numbers
    article_numbers = df['Article Number'].str.replace('Art.-Nr. ', '', regex=False).str.strip()
    
    # Prepare products for LLM extraction
    products = []
    for _, row in df.iterrows():
        products.append({
            "headline": row.get("Headline", ""),
            "short_description": row.get("Short Description", ""),
            "long_description": row.get("Long Description", ""),
            "technical_info": row.get("Technical Information", ""),
            "breadcrumbs": row.get("Breadcrumbs", ""),
        })
    
    # Extract features using LLM
    logger.info(f"Extracting features with LLM ({max_workers_llm} workers)...")
    features_list = extract_product_features_batch(products, max_workers=max_workers_llm)
    
    # Fetch images if requested
    if fetch_images:
        logger.info(f"Fetching product images ({max_workers_images} workers)...")
        urls = df['URL'].tolist()
        image_urls = fetch_images_parallel(urls, max_workers=max_workers_images)
    else:
        image_urls = [None] * len(df)
    
    # Build result DataFrame
    result = pd.DataFrame({
        'article_number': article_numbers,
        'headline': df['Headline'].fillna(''),
        'long_description': df['Long Description'],
        'image_url': image_urls,
        'standard': [f.standard for f in features_list],
        'materials': [f.materials for f in features_list],
        'size': [f.size for f in features_list],
        'dimensions': [f.dimensions for f in features_list],
        'category': [f.category for f in features_list],
        'brand': [f.brand for f in features_list],
        'application': [f.application for f in features_list],
        'additional_info': [f.additional_info for f in features_list],
    })
    
    logger.info(f"Processing complete. {len(result)} products processed.")
    return result


async def process_products_async(
    df: pd.DataFrame,
    max_workers_llm: int = 5,
    max_workers_images: int = 10,
    fetch_images: bool = True
) -> pd.DataFrame:
    """
    Async wrapper for process_products.
    """
    return await asyncio.to_thread(
        process_products, df, max_workers_llm, max_workers_images, fetch_images
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("Data/tools_master.csv")
    
    # Process a small sample first
    sample_df = df.head(3)
    result = process_products(sample_df, fetch_images=False)
    
    print("\n=== Processed Products ===")
    for _, row in result.iterrows():
        print(f"\nArticle: {row['article_number']}")
        print(f"Headline: {row['headline'][:80]}...")
        print(f"Standard: {row['standard']}")
        print(f"Materials: {row['materials']}")
        print(f"Size: {row['size']}")
        print(f"Dimensions: {row['dimensions']}")
        print(f"Category: {row['category']}")
        print(f"Brand: {row['brand']}")
        print(f"Application: {row['application']}")
        print(f"Additional: {row['additional_info']}")
