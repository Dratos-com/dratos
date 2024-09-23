from typing import Dict, List
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
        try:
            table = self.db.open_table(self.table_name)
        except FileNotFoundError as exc:
            raise ValueError(f"Conversation {conversation_id} not found") from exc

        # Find the message to edit
        message = table.search(query=f"id == '{message_id}'").to_pandas()
        if message.empty:
            raise ValueError(f"Message {message_id} not found in conversation {conversation_id}")

        # Update the message
        updated_message = message.iloc[0].to_dict()
        updated_message['content'] = new_content
        updated_message['timestamp'] = datetime.now()

        # Delete the old message and add the updated one
        table.delete(f"id == '{message_id}'")
        table.add([updated_message])

        return updated_message

    def get_all_memories(self):
        return self.table.to_pandas().to_dict('records')

    def get_conversation_memories(self, conversation_id):
        memories = self.table.to_pandas()[self.table.to_pandas()['conversation_id'] == conversation_id]
        return memories.to_dict('records')

    def update_conversation_memories(self, conversation_id: str, conversation: List[Dict]) -> None:
        try:
            table = self.db.open_table(self.table_name)
        except FileNotFoundError:
            self._create_table(self.table_name)
            table = self.db.open_table(self.table_name)
        
        # Clear existing data
        table.delete(f"conversation_id == '{conversation_id}'")
        
        # Add updated conversation data
        data = self._conversation_to_data(conversation_id, conversation)
        table.add(data)

    def _create_table(self, table_name):
        self.db.create_table(table_name, schema=self.schema)

    def _conversation_to_data(self, conversation_id, conversation):
        return [
            {
                "conversation_id": conversation_id,
                "id": message["id"],
                "content": message["content"],
                "sender": message["sender"],
                "timestamp": datetime.fromisoformat(message["timestamp"])
            }
            for message in conversation
        ]
