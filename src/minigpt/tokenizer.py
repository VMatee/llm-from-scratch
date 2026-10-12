"""Character vocabulary with transparent encoding, no pretrained tokenizer."""

class CharTokenizer:
    def __init__(self, vocabulary: list[str]):
        chars = sorted(set(vocabulary))
        if not chars:
            raise ValueError("Vocabulary cannot be empty")
        if any(len(char) != 1 for char in chars):
            raise ValueError("Tokenizer entries must each be one character")
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(list(text))

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        unknown = [c for c in text if c not in self.stoi]
        if unknown:
            raise ValueError(f"Character not in training vocabulary: {unknown[0]!r}")
        return [self.stoi[c] for c in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.chars[i] for i in token_ids)
