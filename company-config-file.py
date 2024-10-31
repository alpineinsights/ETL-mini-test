"""
Configuration file containing the list of company names for the prompt system.
This file can be updated independently of the main application code.
"""

COMPANY_NAMES = [
    "3i Group Plc",
    "3M Company",
    "A. O. Smith Corporation",
    "A.P. Møller - Mærsk A/S",
    "A2A S.p.A.",
    # Add all company names here...
    "Zurich Insurance Group AG"
]

def get_company_names_prompt() -> str:
    """
    Returns formatted company names for use in the system prompt.
    
    Returns:
        str: Newline-separated list of company names
    """
    return "\n".join(COMPANY_NAMES)

def validate_company_names() -> bool:
    """
    Validates that company names are properly formatted.
    Checks for empty strings and proper string type.
    
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If any company name is invalid
    """
    if not COMPANY_NAMES:
        raise ValueError("Company names list cannot be empty")
        
    for company in COMPANY_NAMES:
        if not isinstance(company, str):
            raise ValueError(f"Invalid company name type: {type(company)}")
        if len(company.strip()) == 0:
            raise ValueError("Company name cannot be empty")
            
    # Check for duplicates
    if len(COMPANY_NAMES) != len(set(COMPANY_NAMES)):
        raise ValueError("Duplicate company names found")
        
    return True

# Validate company names when module is imported
validate_company_names()
