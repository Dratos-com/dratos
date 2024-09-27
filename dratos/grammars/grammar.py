from lark import Lark, Transformer

grammar_spec = """
    start: agent_comm
    agent_comm: "Agent" NAME "sends a message to Agent" NAME ":" "Message Content:" "Task:" TASK "Data:" DATA
    %import common.CNAME -> NAME
    %import common.WORD
    TASK: WORD+
    DATA: WORD+
    %ignore " "
"""

class GrammarParser:
    def __init__(self):
        self.parser = Lark(grammar_spec, start='start')

    def parse_communication(self, communication: str):
        return self.parser.parse(communication)

class Grammar:
    def __init__(self, name: str, rules: List[str], components: List[Any]):
        self.name = name
        self.rules = rules
        self.components = components
        self.parser = GrammarParser()

    def apply_rules(self, data: Any) -> Any:
        parsed = self.parser.parse_communication(data["communication"])
        # Apply parsed rules to data
        return data  # Modify as needed