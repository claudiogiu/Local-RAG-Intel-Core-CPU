from enum import Enum


class RetrievalStatus(str, Enum):
    """
    Enumeration defining the possible outcomes of a retrieval operation.
    """

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class PipelineStatus(str, Enum):
    """
    Enumeration defining the possible outcomes of a retrieval operation.
    """

    COMPLETED = "COMPLETED"
    ERROR = "ERROR"