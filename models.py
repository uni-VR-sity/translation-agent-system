import requests
from typing import List, Dict, Optional


class OpenRouterModelsFetcher:
    def __init__(self, free_only: bool = False):
        self.api_url = "https://openrouter.ai/api/v1/models"
        self.free_only = free_only
        self.models = []
    
    
    def fetch_api_alternative(self) -> List[str]:
        """
        Check if OpenRouter has a public API endpoint for models.
        """
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
        
        
        try:
            response = requests.get(self.api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Extract model names from API response
                self.models = []
                if isinstance(data, dict) and 'data' in data:
                    for model in data['data']:
                        if isinstance(model, dict) and 'id' in model:
                            if self.free_only:
                                if model['id'].endswith(':free'):
                                    self.models.append(model['id'])
                            else:
                                self.models.append(model['id'])
                
                return self.models
                
        except Exception as e:
            print(f"API endpoint failed: {e}")
        
        return []
    
    def write_models_to_file(self, filename: str="./data/openrouter_models.txt") -> None:
        """
        Write the fetched models to a file.
        
        Args:
            filename: The name of the file to write to
        """
        try:
            with open(filename, 'w') as f:
                for model in self.models:
                    f.write(f"{model}\n")
            print(f"Models written to {filename}")
        except Exception as e:
            print(f"Error writing to file: {e}")

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

    def write_models_to_file(self, filename: str = "./data/ollama_models.txt") -> None:
        """
        Write the fetched models to a file.
        
        Args:
            filename: The name of the file to write to
        """
        try:
            with open(filename, 'w') as f:
                for model in self.get_model_names():
                    f.write(f"{model}\n")
            print(f"Models written to {filename}")
        except Exception as e:
            print(f"Error writing to file: {e}")


def fetch_openrouter_models(free_only: bool = False) -> List[str]:
    """
    Convenience function to fetch top 5 OpenRouter model names.
    
    Args:
        free_only: If True, only fetch free models (max_price=0)
        order_filter: Sorting filter (e.g., 'top-weekly', 'pricing-low-to-high', etc.)
    """
    fetcher = OpenRouterModelsFetcher(free_only=free_only)
    models = fetcher.fetch_api_alternative()
    if models:
        fetcher.write_models_to_file()
    else:
        print("No models found or API endpoint is not accessible.")
    return models

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
    
    models = fetcher.list_local_models()
    if models:
        fetcher.write_models_to_file()

    return fetcher.get_model_names()

if __name__ == "__main__":
    # Example usage
    print("Fetching OpenRouter models...")
    openrouter_models = fetch_openrouter_models(free_only=False)
    for model in openrouter_models:
        print(model)

    print("\nListing local Ollama models...")
    ollama_models = list_ollama_models(ollama_host="http://localhost:11434", detailed=True)
    for model in ollama_models:
        print(model)

