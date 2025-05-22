from typing import List, Optional
import os
import torch

# Please run `pip install -U sentence-transformers`
from sentence_transformers import SentenceTransformer
from torch import Tensor


class TextEmbedding:
    def __init__(self, embed_type, modelPath=None, device: Optional[torch.device] = None, ):
        if embed_type == 'mpnet':
            embed_type = "all-mpnet-base-v2"
        else:
            embed_type = "average_word_embeddings_glove.6B.300d"

        if modelPath is not None:
            modelPath = os.path.join(modelPath, embed_type)

        if modelPath is not None and os.path.exists(modelPath):
            self.model = SentenceTransformer(modelPath, device=device, )
        else:
            # download from https://hf-mirror.com/ if unable to connect to huggingface
            self.model = SentenceTransformer(f"sentence-transformers/{embed_type}", device=device, )
            self.model.save(modelPath)

    def __call__(self, sentences: List[str]) -> Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)