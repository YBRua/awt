from dataclasses import dataclass
from typing import List


@dataclass
class DataInstance:
    source_code: str
    raw_source_tokens: List[str]
    tokens: List[str]

    def serialize(self):
        return {
            'source_code': self.source_code,
            'raw_source_tokens': self.raw_source_tokens,
            'tokens': self.tokens
        }

    @classmethod
    def deserialize(cls, json_obj):
        return cls(json_obj['source_code'], json_obj['raw_source_tokens'],
                   json_obj['tokens'])
