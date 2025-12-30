import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


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
        print(f"Error fetching {page_url}: {e}")
        return None


def process_products(
    df: pd.DataFrame,
    max_workers: int = 10,
    fetch_images: bool = True
) -> pd.DataFrame:
    """
    Process product data: combine descriptions and extract image URLs.
    
    Args:
        df: Input dataframe with product data
        max_workers: Number of parallel threads for fetching images
        fetch_images: Whether to fetch image URLs (can be slow for large datasets)
        
    Returns:
        DataFrame with columns:
        - article_number: Product article number
        - combined_description: Headline + Short Description + Long Description
        - long_description: Just the long description
        - image_url: URL of the product image
    """
    # Create combined description
    combined_description = (
        df['Headline'].fillna('') + ' ' +
        df['Short Description'].fillna('') + ' ' +
        df['Long Description'].fillna('')
    ).str.strip()
    
    # Initialize result dataframe
    result = pd.DataFrame({
        'article_number': df['Article Number'].str.split('Art.-Nr. ').str[1],
        'combined_description': combined_description,
        'long_description': df['Long Description'],
        'image_url': None
    })
    
    if fetch_images:
        # Fetch image URLs in parallel
        urls = df['URL'].tolist()
        image_urls = [None] * len(urls)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(extract_image_url, url): idx 
                for idx, url in enumerate(urls) if pd.notna(url)
            }
            
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Fetching images"):
                idx = future_to_idx[future]
                image_urls[idx] = future.result()
        
        result['image_url'] = image_urls
    
    return result


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("Data/tools_master.csv")
    
    # Process a small sample first
    sample_df = df.head(5)
    result = process_products(sample_df, fetch_images=True)
    print(result)

