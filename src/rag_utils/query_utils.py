"""
Query utilities for product search.

Provides:
- Query expansion with structured feature extraction (matching product format)
- LLM-based reranking of search results
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from std_utils import get_logger

logger = get_logger("comini.query")


# ============================================================================
# Structured Query Model (matches ProductFeatures format)
# ============================================================================

class StructuredQuery(BaseModel):
    """
    Structured query with the same categories as product features.
    This ensures query format matches the data format for better semantic search.
    """
    
    original_query: str = Field(
        description="The original user query as provided"
    )
    
    # Same categories as ProductFeatures
    standard: str | None = Field(
        default=None,
        description="Industry standards mentioned or implied (e.g., DIN, ISO, VDE). Leave null if none."
    )
    
    materials: str | None = Field(
        default=None,
        description="Materials, coatings, or surface treatments requested (e.g., 'galvanized steel', 'stainless steel A2', 'chrome plated'). Expand abbreviations."
    )
    
    size: str | None = Field(
        default=None,
        description="Size specifications (e.g., 'Torx TX25', 'M6 thread', '1/4 inch drive', 'SW13'). Expand abbreviations."
    )
    
    dimensions: str | None = Field(
        default=None,
        description="Physical dimensions if specified (e.g., 'Length 100mm', 'Diameter 6mm')."
    )
    
    category: str | None = Field(
        default=None,
        description="Product category/type (e.g., 'Screwdriver Bit', 'Hex Nut', 'Socket Wrench', 'Hammer')."
    )
    
    brand: str | None = Field(
        default=None,
        description="Brand or manufacturer if specified (e.g., 'ZEBRA', 'Halder', 'Würth')."
    )
    
    application: str | None = Field(
        default=None,
        description="Intended use or application if mentioned (e.g., 'for automotive', 'for impact drivers')."
    )
    
    additional_info: str | None = Field(
        default=None,
        description="Any other relevant details from the query not covered above."
    )
    
    detected_abbreviations: list[str] = Field(
        default_factory=list,
        description="List of abbreviations that were detected and expanded"
    )


# Legacy model for backwards compatibility
class ExpandedQuery(BaseModel):
    """Structured output for query expansion (legacy format)."""
    original_query: str
    expanded_query: str
    detected_abbreviations: list[str] = Field(default_factory=list)
    language: str = Field(default="de")


# ============================================================================
# Query Expansion Prompt
# ============================================================================

STRUCTURED_QUERY_PROMPT = """You are a query parser for an industrial tools and fasteners catalog from Würth Industrie.

Your task is to parse a user's search query and extract structured information into specific categories.
This structured format will be used for semantic search against a product database.

## Product Context
The catalog contains industrial tools and fasteners including:
- Bits (screwdriver bits): TX (Torx), PH (Phillips), PZ (Pozidriv), SL (Slotted)
- Screws, nuts, bolts, washers
- Hand tools, power tools, measuring instruments
- Socket wrenches, impact tools
- Fastening systems, anchors, rivets

## Abbreviation Reference (expand these)

### Screw Drive Types
- TX → Torx
- PH → Phillips  
- PZ → Pozidriv
- SL → Schlitz (Slotted)
- HX / SW → Sechskant (Hexagon)
- INB / IHX → Innensechskant (Allen/Hex socket)

### Materials & Coatings
- gv zn / galv. verzinkt → galvanisch verzinkt (galvanized)
- A2 / A4 → Edelstahl rostfrei (stainless steel)
- ST → Stahl (Steel)
- MS → Messing (Brass)
- CU → Kupfer (Copper)
- znfl → Zinkflake (Zinc flake coating)
- dacro → Dacromet coating

### Sizes & Measurements
- M4, M5, M6... → Metrisches Gewinde (Metric thread)
- SW → Schlüsselweite (wrench size)
- L → Länge (Length)
- D / Ø → Durchmesser (Diameter)
- ZO → Zoll (inch)

### Product Types
- SKS → Senkschraube (Countersunk screw)
- ZYL → Zylinderschraube (Cylinder head screw)
- SHR → Scheibe (Washer)
- MU → Mutter (Nut)
- SORT → Sortiment (Assortment/Set)
- TLG → Teilig (pieces)

## Instructions
1. Parse the query into the structured categories
2. Expand ALL abbreviations to their full form
3. Only fill in categories that are explicitly mentioned or clearly implied
4. Keep the language consistent (German or English based on query)
5. Leave fields as null if no relevant information is in the query

## Categories to Extract
- **standard**: Industry standards (DIN, ISO, VDE, etc.)
- **materials**: Materials and coatings (steel type, surface treatment)
- **size**: Size specifications (drive size, thread size, key size)
- **dimensions**: Physical measurements (length, diameter, weight)
- **category**: Product type (bit, screw, wrench, hammer, etc.)
- **brand**: Manufacturer/brand name
- **application**: Intended use
- **additional_info**: Any other relevant details"""


def get_query_expander() -> ChatOpenAI:
    """Get the LLM model configured for query expansion."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )


def expand_query_structured(query: str) -> StructuredQuery:
    """
    Parse a user query into structured categories matching the product format.
    
    Args:
        query: The user's search query (may contain abbreviations)
        
    Returns:
        StructuredQuery with parsed categories
    """
    logger.info(f"Parsing query into structured format: {query}")
    
    llm = get_query_expander()
    structured_llm = llm.with_structured_output(StructuredQuery)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", STRUCTURED_QUERY_PROMPT),
        ("human", "Parse this search query: {query}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"query": query})
    
    # Log extracted features
    logger.info(f"Parsed query - Category: {result.category}, Size: {result.size}, Materials: {result.materials}")
    if result.detected_abbreviations:
        logger.info(f"Expanded abbreviations: {result.detected_abbreviations}")
    
    return result


def build_search_text_from_query(structured: StructuredQuery) -> str:
    """
    Build a search text from structured query that matches the product embedding format.
    
    Includes:
    1. Original query text (like product headline)
    2. Structured features (like product extracted features)
    
    Args:
        structured: The structured query with parsed categories
        
    Returns:
        Formatted text for embedding/search
    """
    parts = []
    
    # Include original query text (matches product headline/description)
    if structured.original_query:
        parts.append(structured.original_query)
    
    # Add structured features in the same format as products
    feature_fields = [
        ('Category', structured.category),
        ('Size', structured.size),
        ('Dimensions', structured.dimensions),
        ('Materials', structured.materials),
        ('Standard', structured.standard),
        ('Brand', structured.brand),
        ('Application', structured.application),
        ('Additional', structured.additional_info),
    ]
    
    for label, value in feature_fields:
        # Only include non-null, non-empty values
        if value is not None and str(value).strip():
            parts.append(f"{label}: {value}")
    
    return ' '.join(parts)


def expand_query(query: str) -> ExpandedQuery:
    """
    Expand a user query by parsing into structured format and building search text.
    
    Maintains backwards compatibility with the ExpandedQuery format.
    
    Args:
        query: The user's search query (may contain abbreviations)
        
    Returns:
        ExpandedQuery with original and expanded versions
    """
    # Parse into structured format
    structured = expand_query_structured(query)
    
    # Build the expanded query text matching product format
    expanded_text = build_search_text_from_query(structured)
    
    logger.info(f"Expanded query: {expanded_text}")
    
    return ExpandedQuery(
        original_query=query,
        expanded_query=expanded_text,
        detected_abbreviations=structured.detected_abbreviations,
        language="de"  # Default to German
    )


async def expand_query_async(query: str) -> ExpandedQuery:
    """Async version of expand_query."""
    import asyncio
    return await asyncio.to_thread(expand_query, query)


async def expand_query_structured_async(query: str) -> StructuredQuery:
    """Async version of expand_query_structured."""
    import asyncio
    return await asyncio.to_thread(expand_query_structured, query)


# ============================================================================
# LLM Reranking
# ============================================================================

class RerankResult(BaseModel):
    """Result from LLM reranking of search results."""
    
    best_match_index: int = Field(
        description="The 0-based index of the best matching product from the list"
    )
    reasoning: str = Field(
        description="Brief explanation of why this is the best match"
    )
    confidence: str = Field(
        description="Confidence level: 'high', 'medium', or 'low'"
    )



RERANK_PROMPT = """You are an expert at matching product search queries to industrial tool and fastener products.

Given a search query and a list of candidate products with their extracted features, identify which product is the BEST match.

## Query
{query}

## Candidate Products (0-indexed)
{products}

## Instructions
1. Analyze the query to understand what product is being requested
2. Consider all product features: category, size, dimensions, materials, standard, brand, application
3. Match specific requirements: size specifications, material types, coating, drive types
4. Pick the single best match from the list
5. If multiple seem equally good, prefer exact specification matches

Return the index (0-based) of the best matching product, your confidence level, and brief reasoning."""


def format_product_for_rerank(idx: int, product: dict) -> str:
    """Format a product dict for the reranking prompt."""
    lines = [f"[{idx}] Article: {product.get('article_number', 'N/A')}"]
    
    # Add headline
    if product.get('headline'):
        lines.append(f"    Headline: {product['headline'][:100]}")
    
    # Add structured features
    features = [
        ('Category', 'category'),
        ('Size', 'size'),
        ('Dimensions', 'dimensions'),
        ('Materials', 'materials'),
        ('Standard', 'standard'),
        ('Brand', 'brand'),
        ('Application', 'application'),
    ]
    
    for label, key in features:
        value = product.get(key)
        if value:
            lines.append(f"    {label}: {value}")
    
    # Add description snippet
    if product.get('long_description'):
        desc = product['long_description'][:150]
        lines.append(f"    Description: {desc}...")
    
    return "\n".join(lines)


def rerank_results(
    query: str,
    products: list[dict],
) -> RerankResult:
    """
    Use LLM to identify the best match from a list of products.
    
    Args:
        query: The original search query
        products: List of product dicts with article_number, headline, and LLM-extracted features
        
    Returns:
        RerankResult with the index of the best match
    """
    if not products:
        return RerankResult(
            best_match_index=0,
            confidence="low",
            reasoning="No products to rank"
        )
    
    logger.info(f"Reranking {len(products)} results for query: {query}")
    
    # Format products for the prompt with all features
    products_text = "\n\n".join([
        format_product_for_rerank(i, p)
        for i, p in enumerate(products)
    ])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(RerankResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert product matcher for industrial tools and fasteners."),
        ("human", RERANK_PROMPT)
    ])
    
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "query": query,
        "products": products_text
    })
    
    # Validate index is in range
    if result.best_match_index < 0 or result.best_match_index >= len(products):
        logger.warning(f"LLM returned invalid index {result.best_match_index}, defaulting to 0")
        result.best_match_index = 0
    
    logger.info(f"Best match: index {result.best_match_index} ({result.confidence} confidence)")
    logger.info(f"Reasoning: {result.reasoning}")
    
    return result


async def rerank_results_async(
    query: str,
    products: list[dict],
) -> RerankResult:
    """Async version of rerank_results."""
    import asyncio
    return await asyncio.to_thread(rerank_results, query, products)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test the structured query expansion
    test_queries = [
        "TX25 bit",
        "M6 gv zn schraube",
        "PH2 schraubendreher 100mm",
        "6kt mutter A2",
        "SORT 14-TLG bits",
        "1/2 zoll steckschlüssel SW17",
    ]
    
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Original: {q}")
        
        # Structured format
        structured = expand_query_structured(q)
        print(f"\nStructured:")
        print(f"  Category: {structured.category}")
        print(f"  Size: {structured.size}")
        print(f"  Materials: {structured.materials}")
        print(f"  Dimensions: {structured.dimensions}")
        print(f"  Standard: {structured.standard}")
        print(f"  Brand: {structured.brand}")
        print(f"  Application: {structured.application}")
        print(f"  Additional: {structured.additional_info}")
        print(f"  Abbreviations: {structured.detected_abbreviations}")
        
        # Search text
        search_text = build_search_text_from_query(structured)
        print(f"\nSearch text: {search_text}")
