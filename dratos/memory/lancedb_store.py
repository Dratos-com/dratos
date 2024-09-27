from typing import Dict, List, Any
import lancedb
import pyarrow as pa
from datetime import datetime

class LanceDBMemoryStore:
    def __init__(self, uri="lancedb"):
        self.db = lancedb.connect(uri)
        self.memories_table_name = "memories"
        self.posts_table_name = "posts"
        
        # Define the schema for our memories table
        self.memories_schema = pa.schema([
            ('conversation_id', pa.string()),
            ('id', pa.string()),
            ('content', pa.string()),
            ('sender', pa.string()),
            ('timestamp', pa.timestamp('us'))
        ])

        # Define the schema for our posts table
        self.posts_schema = pa.schema([
            ('id', pa.string()),
            ('title', pa.string()),
            ('content', pa.string()),
            ('author', pa.string()),
            ('timestamp', pa.timestamp('us')),
            ('upvotes', pa.int32()),
            ('downvotes', pa.int32()),
            ('comments', pa.list_(pa.string())),
            ('branches', pa.list_(pa.string())),
            ('parent_id', pa.string()),  # New field for tracking post lineage
            ('version', pa.int32())  # New field for versioning
        ])

        # Create the memories table if it doesn't exist
        if self.memories_table_name not in self.db.table_names():
            self.memories_table = self.db.create_table(self.memories_table_name, schema=self.memories_schema)
        else:
            self.memories_table = self.db.open_table(self.memories_table_name)

        # Create the posts table if it doesn't exist
        if self.posts_table_name not in self.db.table_names():
            self.posts_table = self.db.create_table(self.posts_table_name, schema=self.posts_schema)
        else:
            self.posts_table = self.db.open_table(self.posts_table_name)

    def add_memory(self, conversation_id, message):
        try:
            # Ensure timestamp is in the correct format
            timestamp = datetime.fromisoformat(message["timestamp"])
            
            self.memories_table.add([{
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
            table = self.db.open_table(self.memories_table_name)
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
        return self.memories_table.to_pandas().to_dict('records')

    def get_conversation_memories(self, conversation_id):
        memories = self.memories_table.to_pandas()[self.memories_table.to_pandas()['conversation_id'] == conversation_id]
        return memories.to_dict('records')

    def update_conversation_memories(self, conversation_id: str, conversation: List[Dict]) -> None:
        try:
            table = self.db.open_table(self.memories_table_name)
        except FileNotFoundError:
            self._create_table(self.memories_table_name)
            table = self.db.open_table(self.memories_table_name)
        
        # Clear existing data
        table.delete(f"conversation_id == '{conversation_id}'")
        
        # Add updated conversation data
        data = self._conversation_to_data(conversation_id, conversation)
        table.add(data)

    def _create_table(self, table_name):
        self.db.create_table(table_name, schema=self.memories_schema)

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

    def add_post(self, post: Dict[str, Any]):
        try:
            # Ensure timestamp is in the correct format
            if isinstance(post['timestamp'], str):
                post['timestamp'] = datetime.fromisoformat(post['timestamp'])
            
            self.posts_table.add([post])
        except (TypeError, ValueError) as e:
            print(f"Error adding post: {e}")
            raise

    def get_post(self, post_id):
        result = self.posts_table.search(query=f"id == '{post_id}'").to_pandas()
        if result.empty:
            return None
        return result.iloc[0].to_dict()

    def update_post(self, post_id, updates):
        post = self.get_post(post_id)
        if not post:
            raise ValueError(f"Post {post_id} not found")

        # Increment the version
        updates['version'] = post['version'] + 1

        # Create a new entry with updated fields
        new_post = {**post, **updates}
        new_post['id'] = f"{post_id}_v{new_post['version']}"

        self.add_post(new_post)

    def get_post_history(self, post_id):
        base_id = post_id.split('_v')[0]
        history = self.posts_table.search(query=f"id LIKE '{base_id}%'").to_pandas()
        return history.sort_values('version').to_dict('records')

    def create_branch(self, parent_id, new_post_data):
        parent_post = self.get_post(parent_id)
        if not parent_post:
            raise ValueError(f"Parent post {parent_id} not found")

        new_post = {
            **new_post_data,
            "parent_id": parent_id,
            "version": 1,
            "branches": [],
            "comments": [],
            "upvotes": 0,
            "downvotes": 0
        }
        self.add_post(new_post)

        # Update parent post to include new branch
        parent_post['branches'].append(new_post['id'])
        self.update_post(parent_id, {"branches": parent_post['branches']})

        return new_post

    def merge_posts(self, source_id, target_id):
        source_post = self.get_post(source_id)
        target_post = self.get_post(target_id)

        if not source_post or not target_post:
            raise ValueError("Source or target post not found")

        # Create a new merged post
        merged_post = {
            **target_post,
            "content": f"{target_post['content']}\n\nMerged with {source_id}:\n{source_post['content']}",
            "comments": target_post['comments'] + source_post['comments'],
            "branches": list(set(target_post['branches'] + source_post['branches'] + [source_id])),
            "version": target_post['version'] + 1
        }

        self.update_post(target_id, merged_post)
        return merged_post

    def get_all_posts(self) -> List[Dict[str, Any]]:
        try:
            posts = self.posts_table.to_pandas()
            # Convert datetime objects to ISO format strings
            posts['timestamp'] = posts['timestamp'].apply(lambda x: x.isoformat())
            return posts.to_dict('records')
        except Exception as e:
            print(f"Error fetching posts: {e}")
            return []

    def get_recent_posts(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self.posts_table.to_pandas().sort_values('timestamp', ascending=False).head(limit).to_dict('records')
