import requests
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup


class OpenRouterModelsFetcher:
    def __init__(self):
        self.url = "https://openrouter.ai/models"
        self.models = []
    
    def fetch_with_selenium(self) -> List[str]:
        """
        Fetch models using Selenium WebDriver to handle JavaScript rendering.
        Returns top 5 model names.
        """
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        driver = None
        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(self.url)
            
            # Wait for models to load (adjust timeout as needed)
            wait = WebDriverWait(driver, 15)
            
            # Try different selectors that might contain model names
            possible_selectors = [
                "[data-testid*='model']",
                ".model-card",
                ".model-name",
                "[class*='model']",
                "h3", "h4",  # Model names might be in headers
                ".flex .font-semibold",  # Common pattern for model names
            ]
            
            models = []
            for selector in possible_selectors:
                try:
                    elements = wait.until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector))
                    )
                    
                    for element in elements[:10]:  # Get more than 5 to filter
                        text = element.text.strip()
                        if text and len(text) > 2:  # Filter out empty or very short text
                            models.append(text)
                    
                    if models:
                        break
                        
                except TimeoutException:
                    continue
            
            # If no specific selectors worked, try general approach
            if not models:
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')
                
                # Look for text patterns that might be model names
                potential_models = []
                for text in soup.get_text().split('\n'):
                    text = text.strip()
                    # Filter for potential model names (adjust criteria as needed)
                    if (text and 
                        len(text) > 5 and 
                        len(text) < 50 and
                        ('gpt' in text.lower() or 
                         'claude' in text.lower() or
                         'llama' in text.lower() or
                         'mistral' in text.lower() or
                         'gemini' in text.lower() or
                         'model' in text.lower())):
                        potential_models.append(text)
                
                models = potential_models
            
            # Clean and deduplicate models
            cleaned_models = []
            seen = set()
            for model in models:
                # Clean model name
                clean_name = model.replace('\n', ' ').strip()
                if clean_name not in seen and len(clean_name) > 2:
                    cleaned_models.append(clean_name)
                    seen.add(clean_name)
            
            return cleaned_models[:5]
            
        except WebDriverException as e:
            print(f"WebDriver error: {e}")
            return []
        finally:
            if driver:
                driver.quit()
    
    def fetch_with_requests_session(self) -> List[str]:
        """
        Attempt to fetch models using requests with headers that mimic a browser.
        This might work if the site doesn't heavily rely on JavaScript for initial content.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        try:
            session = requests.Session()
            response = session.get(self.url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to extract model names from the HTML
            models = []
            
            # Look for common patterns where model names might appear
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'span', 'div']):
                text = element.get_text().strip()
                if (text and 
                    len(text) > 5 and 
                    len(text) < 50 and
                    any(keyword in text.lower() for keyword in ['gpt', 'claude', 'llama', 'mistral', 'gemini'])):
                    models.append(text)
            
            return list(dict.fromkeys(models))[:5]  # Remove duplicates and get top 5
            
        except Exception as e:
            print(f"Requests error: {e}")
            return []
    
    def fetch_api_alternative(self) -> List[str]:
        """
        Check if OpenRouter has a public API endpoint for models.
        """
        api_endpoints = [
            "https://openrouter.ai/api/v1/models",
            "https://api.openrouter.ai/api/v1/models",
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
        
        for endpoint in api_endpoints:
            try:
                response = requests.get(endpoint, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract model names from API response
                    models = []
                    if isinstance(data, dict) and 'data' in data:
                        for model in data['data'][:5]:
                            if isinstance(model, dict) and 'id' in model:
                                models.append(model['id'])
                    elif isinstance(data, list):
                        for model in data[:5]:
                            if isinstance(model, dict) and 'id' in model:
                                models.append(model['id'])
                    
                    return models
                    
            except Exception as e:
                print(f"API endpoint {endpoint} failed: {e}")
                continue
        
        return []
    
    def get_top_5_models(self) -> List[str]:
        """
        Main method to fetch top 5 model names from OpenRouter.
        Tries multiple approaches in order of preference.
        """
        print("Attempting to fetch OpenRouter models...")
        
        # Method 1: Try API first (fastest and most reliable)
        models = self.fetch_api_alternative()
        if models:
            print(f"Successfully fetched {len(models)} models via API")
            return models
        
        # Method 2: Try requests with browser headers
        models = self.fetch_with_requests_session()
        if models:
            print(f"Successfully fetched {len(models)} models via requests")
            return models
        
        # Method 3: Use Selenium as fallback
        try:
            models = self.fetch_with_selenium()
            if models:
                print(f"Successfully fetched {len(models)} models via Selenium")
                return models
        except Exception as e:
            print(f"Selenium method failed: {e}")
        
        # If all methods fail, return a default list
        print("All methods failed, returning fallback models")
        return [
            "anthropic/claude-3-opus",
            "openai/gpt-4-turbo",
            "google/gemini-pro",
            "meta-llama/llama-2-70b-chat",
            "mistralai/mistral-7b-instruct"
        ]


class OllamaModelsFetcher:
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        """
        Initialize the Ollama models fetcher.
        
        Args:
            ollama_host: The Ollama API endpoint (default: http://localhost:11434)
        """
        self.ollama_host = ollama_host.rstrip('/')
        self.api_endpoint = f"{self.ollama_host}/api"
    
    def is_ollama_running(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Ollama not accessible: {e}")
            return False
    
    def list_local_models(self) -> List[Dict[str, any]]:
        """
        List all locally available Ollama models with detailed information.
        
        Returns:
            List of dictionaries containing model information
        """
        if not self.is_ollama_running():
            print("Error: Ollama is not running or not accessible")
            return []
        
        try:
            response = requests.get(f"{self.api_endpoint}/tags", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = data.get('models', [])
            
            # Sort models by name for consistent output
            models.sort(key=lambda x: x.get('name', ''))
            
            return models
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Ollama models: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []
    
    def get_model_names(self) -> List[str]:
        """
        Get a simple list of model names only.
        
        Returns:
            List of model names
        """
        models = self.list_local_models()
        return [model.get('name', '') for model in models if model.get('name')]
    
    def get_model_details(self, model_name: str) -> Optional[Dict[str, any]]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model to get details for
            
        Returns:
            Dictionary with model details or None if not found
        """
        models = self.list_local_models()
        for model in models:
            if model.get('name') == model_name:
                return model
        return None
    
    def format_model_info(self, model: Dict[str, any]) -> str:
        """
        Format model information into a readable string.
        
        Args:
            model: Dictionary containing model information
            
        Returns:
            Formatted string with model information
        """
        name = model.get('name', 'Unknown')
        size = model.get('size', 0)
        modified = model.get('modified_at', '')
        
        # Convert size to human readable format
        if size:
            if size >= 1_000_000_000:  # GB
                size_str = f"{size / 1_000_000_000:.1f} GB"
            elif size >= 1_000_000:  # MB
                size_str = f"{size / 1_000_000:.1f} MB"
            else:  # KB
                size_str = f"{size / 1_000:.1f} KB"
        else:
            size_str = "Unknown size"
        
        # Format modified date
        if modified:
            try:
                from datetime import datetime
                modified_dt = datetime.fromisoformat(modified.replace('Z', '+00:00'))
                modified_str = modified_dt.strftime('%Y-%m-%d %H:%M')
            except:
                modified_str = modified
        else:
            modified_str = "Unknown"
        
        return f"{name:<30} | {size_str:<10} | Modified: {modified_str}"
    
    def display_models(self, detailed: bool = True) -> None:
        """
        Display all local Ollama models in a formatted table.
        
        Args:
            detailed: If True, show detailed information; if False, show names only
        """
        models = self.list_local_models()
        
        if not models:
            print("No Ollama models found or Ollama is not running.")
            return
        
        print(f"Found {len(models)} local Ollama model(s):")
        print("-" * 70)
        
        if detailed:
            print(f"{'Model Name':<30} | {'Size':<10} | {'Modified'}")
            print("-" * 70)
            for model in models:
                print(self.format_model_info(model))
        else:
            for i, model in enumerate(models, 1):
                print(f"{i}. {model.get('name', 'Unknown')}")
        
        print("-" * 70)



def fetch_openrouter_models() -> List[str]:
    """
    Convenience function to fetch top 5 OpenRouter model names.
    """
    fetcher = OpenRouterModelsFetcher()
    return fetcher.get_top_5_models()

def list_ollama_models(ollama_host: str = "http://localhost:11434", detailed: bool = True) -> List[str]:
    """
    Convenience function to list local Ollama models.
    
    Args:
        ollama_host: The Ollama API endpoint (default: http://localhost:11434)
        detailed: If True, display detailed info; if False, return names only
    
    Returns:
        List of model names
    """
    fetcher = OllamaModelsFetcher(ollama_host)
    
    if detailed:
        fetcher.display_models(detailed=True)
    
    return fetcher.get_model_names()

def get_all_available_models(ollama_host: str = "http://localhost:11434") -> Dict[str, List[str]]:
    """
    Get both OpenRouter and local Ollama models.
    
    Args:
        ollama_host: Ollama API endpoint
    
    Returns:
        Dictionary with 'openrouter' and 'ollama' model lists
    """
    print("Fetching all available models...")
    
    # Get OpenRouter models
    openrouter_models = fetch_openrouter_models()
    
    # Get local Ollama models
    ollama_fetcher = OllamaModelsFetcher(ollama_host)
    ollama_models = ollama_fetcher.get_model_names()
    
    return {
        'openrouter': openrouter_models,
        'ollama': ollama_models
    }

# Example usage
if __name__ == "__main__":
    # Example 1: List local Ollama models only
    print("=== Local Ollama Models ===")
    ollama_models = list_ollama_models()
    
    # Example 2: Get OpenRouter models from saved HTML
    print("\n=== OpenRouter Models ===")
    # openrouter_models = fetch_openrouter_models("Models _ OpenRouter.html")
    openrouter_models = fetch_openrouter_models()
    
    print("Top 5 OpenRouter models:")
    for i, model in enumerate(openrouter_models, 1):
        print(f"{i}. {model}")
    
    # Example 3: Get all models at once
    print("\n=== All Available Models ===")
    all_models = get_all_available_models()
    
    print(f"\nSummary:")
    print(f"- OpenRouter models: {len(all_models['openrouter'])}")
    print(f"- Local Ollama models: {len(all_models['ollama'])}")
    