import lancedb
import pyarrow as pa
from datetime import datetime

class LanceDBMemoryStore:
    def __init__(self, uri="lancedb"):
        self.db = lancedb.connect(uri)
        self.table_name = "memories"
        
        # Define the schema for our table
        self.schema = pa.schema([
            ('conversation_id', pa.string()),
            ('id', pa.string()),
            ('content', pa.string()),
            ('sender', pa.string()),
            ('timestamp', pa.timestamp('us'))
        ])

        # Create the table if it doesn't exist
        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=self.schema)
        else:
            self.table = self.db.open_table(self.table_name)

    def add_memory(self, conversation_id, message):
        try:
            # Ensure timestamp is in the correct format
            timestamp = datetime.fromisoformat(message["timestamp"])
            
            self.table.add([{
                "conversation_id": conversation_id,
                "id": message["id"],
                "content": message["content"],
                "sender": message["sender"],
                "timestamp": timestamp
            }])
        except (TypeError, ValueError) as e:
            print(f"Error adding memory: {e}")
            raise

    def edit_memory(self, conversation_id, message_id, new_content):
        # This is a simple implementation. In a real-world scenario, you might want to use more efficient update methods.
        memories = self.table.to_pandas()
        memory_index = memories[(memories["conversation_id"] == conversation_id) & (memories["id"] == message_id)].index
        if len(memory_index) > 0:
            self.table.update(memory_index[0], {"content": new_content})
            return {"id": message_id, "content": new_content}
        return None

    def get_all_memories(self):
        return self.table.to_pandas().to_dict('records')

    def get_conversation_memories(self, conversation_id):
        memories = self.table.to_pandas()[self.table.to_pandas()['conversation_id'] == conversation_id]
        return memories.to_dict('records')