"""
Configuration file for system prompts used in the application.
Handles prompt templates and their configuration.
"""

from typing import Optional
from company_config import get_company_names_prompt

# Default context prompt template
DEFAULT_PROMPT_TEMPLATE = """Give a short succinct context to situate this chunk within the overall enclosed document boader context for the purpose of improving similarity search retrieval of the chunk. 

Make sure to list:
1. The name of the main company mentioned AND any other secondary companies mentioned if applicable. ONLY use company names exact spellings from the list below to facilitate similarity search retrieval.
2. The apparent date of the document (YYYY.MM.DD)
3. Any fiscal period mentioned. ALWAYS use BOTH abreviated tags (e.g. Q1 2024, Q2 2024, H1 2024) AND more verbose tags (e.g. first quarter 2024, second quarter 2024, first semester 2024) to improve retrieval.
4. A very succint high level overview (i.e. not a summary) of the chunk's content in no more than 100 characters with a focus on keywords for better similarity search retrieval

Answer only with the succinct context, and nothing else (no introduction, no conclusion, no headings).

List of company names (use exact spelling) : 
{company_names}"""

def get_default_prompt() -> str:
    """
    Returns the complete default prompt with company names.
    
    Returns:
        str: Formatted prompt with company names included
    """
    try:
        return DEFAULT_PROMPT_TEMPLATE.format(
            company_names=get_company_names_prompt()
        )
    except Exception as e:
        raise Exception(f"Error generating default prompt: {str(e)}")

def get_custom_prompt(template: str) -> str:
    """
    Returns a custom prompt with company names.
    
    Args:
        template (str): Custom prompt template with {company_names} placeholder
        
    Returns:
        str: Formatted prompt with company names included
        
    Raises:
        ValueError: If template doesn't contain company_names placeholder
    """
    if "{company_names}" not in template:
        raise ValueError("Custom template must contain {company_names} placeholder")
        
    try:
        return template.format(
            company_names=get_company_names_prompt()
        )
    except Exception as e:
        raise Exception(f"Error generating custom prompt: {str(e)}")
