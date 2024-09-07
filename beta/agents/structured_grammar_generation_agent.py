import daft
from lark import Lark
from typing import Dict, Any
from Beta.beta.data.schema.grammar import Grammar


class StructuredGrammarGenerationAgent:
    def __init__(self):
        self.grammar = None
        self.dataframe = None

    def generate_grammar(self, prompt: str) -> Grammar:
        # TODO: Implement AI-based grammar generation from prompt
        # For now, we'll use a simple example grammar
        grammar = Grammar(name="SimpleGrammar", description="Generated from prompt")
        grammar.add_rule("start", "field+")
        grammar.add_rule("field", "NAME ':' TYPE")
        grammar.add_rule("NAME", "/[a-zA-Z_][a-zA-Z0-9_]*/")
        grammar.add_rule("TYPE", "'string' | 'int' | 'float' | 'boolean'")
        self.grammar = grammar
        return grammar

    def parse_input(self, input_text: str) -> Dict[str, Any]:
        if not self.grammar:
            raise ValueError("Grammar not generated. Call generate_grammar first.")

        parser = Lark(self.grammar.to_lark_grammar(), start="start")
        tree = parser.parse(input_text)

        data = {}
        for field in tree.children:
            name = field.children[0].value
            type_ = field.children[1].value
            data[name] = type_

        return data

    def create_dataframe(self, parsed_data: Dict[str, Any]):
        columns = []
        for name, type_ in parsed_data.items():
            if type_ == "string":
                col = daft.col(name, dtype=daft.DataType.string())
            elif type_ == "int":
                col = daft.col(name, dtype=daft.DataType.int64())
            elif type_ == "float":
                col = daft.col(name, dtype=daft.DataType.float64())
            elif type_ == "boolean":
                col = daft.col(name, dtype=daft.DataType.boolean())
            else:
                raise ValueError(f"Unsupported type: {type_}")
            columns.append(col)

        self.dataframe = daft.DataFrame(columns)
        return self.dataframe

    def process(self, prompt: str, input_text: str) -> daft.DataFrame:
        self.generate_grammar(prompt)
        parsed_data = self.parse_input(input_text)
        return self.create_dataframe(parsed_data)


# Example usage:
# agent = StructuredGrammarGenerationAgent()
# df = agent.process("Create a grammar for user data", "name: string\nage: int\nis_active: boolean")
# print(df.schema)
