from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    """Configuration for text processing pipeline."""
    model_path: str
    data_directory: str
    filename_pattern: str
    batch_size_triple: int = 16
    batch_size_concept: int = 64
    output_directory: str = "./generation_result_debug"
    total_shards_triple: int = 1
    current_shard_triple: int = 0
    total_shards_concept: int = 1
    current_shard_concept: int = 0
    use_8bit: bool = False
    debug_mode: bool = False
    resume_from: int = 0
    record : bool = False
    max_new_tokens: int = 8192
    max_workers: int = 8
    remove_doc_spaces: bool = False