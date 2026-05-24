import logging
from typing import List, Optional
from datasets import load_dataset
from src.config.constants import HF_DATASET_NAME, HF_DATASET_SUBSET

logger = logging.getLogger(__name__)


class Loader:
    """
    Interface for retrieving textual documents from a HuggingFace dataset using 
    environment‑defined configuration parameters for dataset selection and subset specification.

    Attributes:
        dataset_name (str): Name of the HuggingFace dataset to be loaded.
        dataset_subset (str): Name of the dataset subset used for retrieval operations.

    Methods:
        load() -> List[str]:
            Loads the configured HuggingFace dataset subset, extracts textual fields, 
            and returns a list of raw text documents.
    """

    def __init__(self) -> None:
        self.dataset_name: Optional[str] = HF_DATASET_NAME
        self.dataset_subset: Optional[str] = HF_DATASET_SUBSET

    def load(self) -> List[str]:
        if self.dataset_name is None:
            raise ValueError("HF_DATASET_NAME is not defined in the environment.")
        if self.dataset_subset is None:
            raise ValueError("HF_DATASET_SUBSET is not defined in the environment.")

        logger.info(
            f"Loading HuggingFace dataset '{self.dataset_name}' with subset '{self.dataset_subset}'."
        )

        dataset = load_dataset(self.dataset_name, self.dataset_subset, split="train")

        logger.info(f"Dataset loaded with {len(dataset)} entries.")

        documents: List[str] = []
        for entry in dataset:
            text: Optional[str] = entry.get("text")
            if text is not None:
                documents.append(text)

        logger.info(f"Extracted {len(documents)} text documents.")

        return documents
