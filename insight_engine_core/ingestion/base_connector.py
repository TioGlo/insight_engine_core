from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseDataConnector(ABC):
    def __init__(self, source_name: str, config: Dict[str, Any] = None):
        self.source_name = source_name
        self.config = config if config is not None else {}
        print(f"BaseDataConnector '{self.source_name}' initialized.") # For debug

    @abstractmethod
    def fetch_data(self, query: Any = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetches data from the source.
        Returns a list of dictionaries, where each dictionary represents a raw data item.
        Expected keys in dict: 'internal_id', 'content' (which itself can be a dict or string), 'metadata' (optional)
        """
        pass

    def get_source_name(self) -> str:
        return self.source_name
