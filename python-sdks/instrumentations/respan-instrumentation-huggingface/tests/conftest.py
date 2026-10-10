import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    TextGenerationPipeline,
)


@pytest.fixture(scope="session")
def pipeline():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[EOS]": 2,
        "Tracing": 3,
        "Hugging": 4,
        "Face": 5,
        "native": 6,
        "input": 7,
        "output": 8,
        ".": 9,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = '{% for message in messages %}{{ message["role"] + ": " + message["content"] + " " }}{% endfor %}'
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            name_or_path="controlled-local-gpt2",
            n_positions=8192,
            n_ctx=8192,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    model.eval()
    return TextGenerationPipeline(model=model, tokenizer=tokenizer, device=-1)
