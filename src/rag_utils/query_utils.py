from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from std_utils import get_logger

logger = get_logger("comini.query")


# Pydantic model for structured LLM output
class ExpandedQuery(BaseModel):
    """Structured output for query expansion."""
    
    original_query: str = Field(
        description="The original user query as provided"
    )
    expanded_query: str = Field(
        description="The expanded query with all abbreviations replaced with full terms"
    )
    detected_abbreviations: list[str] = Field(
        default_factory=list,
        description="List of abbreviations that were detected and expanded"
    )
    language: str = Field(
        default="de",
        description="Detected language of the query (de/en)"
    )


# System prompt with context about the product catalog
QUERY_EXPANSION_PROMPT = """You are a query expansion assistant for an industrial tools and fasteners catalog from Würth Industrie.

Your task is to:
1. Expand ALL abbreviations to their full form
2. Keep the query in its original language (German or English)
3. Make the query more searchable while preserving the original intent

## Product Context
The catalog contains industrial tools and fasteners including:
- Bits (screwdriver bits): TX (Torx), PH (Phillips), PZ (Pozidriv), SL (Slotted)
- Screws, nuts, bolts, washers
- Hand tools, power tools, measuring instruments
- Fastening systems, anchors, rivets

## Common Abbreviations to Expand

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
- ZO → Zoll-Außensechskant (inch outer hex)

### Product Types
- SKS → Senkschraube (Countersunk screw)
- ZYL → Zylinderschraube (Cylinder head screw)
- LI-SKS → Linsensenkschraube (Oval countersunk screw)
- SHR → Scheibe (Washer)
- MU → Mutter (Nut)
- SORT → Sortiment (Assortment/Set)
- TLG → Teilig (pieces, e.g., 14-TLG = 14-piece)

### Other Common Abbreviations
- LO → Mit Loch/Bohrung (With hole)
- KS → Kunststoff (Plastic)
- RD → Rund (Round)
- 6KT → Sechskant (Hexagon)
- FL → Flach (Flat)
- DPBIT → Doppelbit (Double-ended bit)

## Instructions
- Expand abbreviations contextually (TX in "TX bit" → "Torx bit")
- Keep product codes intact (e.g., "Art.-Nr. 06143106" stays as is)
- Don't add information that wasn't in the original query
- If unsure about an abbreviation, keep it as is

Return the expanded query that would be better for semantic search."""


def get_query_expander() -> ChatOpenAI:
    """Get the LLM model configured for query expansion."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,  # Deterministic output for consistency
    )


def expand_query(query: str) -> ExpandedQuery:
    """
    Expand a user query by replacing abbreviations with full terms.
    
    Uses an LLM to intelligently expand abbreviations in the context
    of industrial tools and fasteners.
    
    Args:
        query: The user's search query (may contain abbreviations)
        
    Returns:
        ExpandedQuery with original and expanded versions
    """
    logger.info(f"Expanding query: {query}")
    
    llm = get_query_expander()
    
    # Use structured output with Pydantic
    structured_llm = llm.with_structured_output(ExpandedQuery)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_EXPANSION_PROMPT),
        ("human", "Expand this query: {query}")
    ])
    
    chain = prompt | structured_llm
    
    result = chain.invoke({"query": query})
    
    logger.info(f"Expanded query: {result.expanded_query}")
    if result.detected_abbreviations:
        logger.info(f"Detected abbreviations: {result.detected_abbreviations}")
    
    return result


async def expand_query_async(query: str) -> ExpandedQuery:
    """
    Async version of expand_query.
    
    Args:
        query: The user's search query
        
    Returns:
        ExpandedQuery with original and expanded versions
    """
    import asyncio
    return await asyncio.to_thread(expand_query, query)


if __name__ == "__main__":
    # Test the query expansion
    test_queries = [
        "TX25 bit",
        "M6 gv zn schraube",
        "PH2 schraubendreher 100mm",
        "6kt mutter A2",
        "SORT 14-TLG bits",
    ]
    
    for q in test_queries:
        result = expand_query(q)
        print(f"\nOriginal: {result.original_query}")
        print(f"Expanded: {result.expanded_query}")
        print(f"Abbreviations: {result.detected_abbreviations}")

