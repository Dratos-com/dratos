class ToolManager:
    def __init__(self, 
        
    )
    def register_tool(self, tool):
        tool_manager.register_tool(tool)
    def get_tool(self, name: str):
        return tool_manager.get_tool(name)

    def list_tools(self):
        return tool_manager.list_tools()
    def execute_tool(self, name: str, *args, session: Session = None, **kwargs):
        return tool_manager.execute_tool(name, *args, session=session, **kwargs)
