from beta.tools.tool import Tool
from beta.integrations.emotiv import EmotivIntegration


class EmotivTool(Tool):
    name = "emotiv"
    description = "Interact with Emotiv EPOC X and process BCI data"

    def __init__(self):
        self.emotiv = EmotivIntegration(config)

    async def execute(self, session_id: str, *args, **kwargs):
        self.emotiv.connect()
        data = await self.emotiv.get_data(session_id)
        processed_data = self.emotiv.process_bci_data(data)
        return processed_data
