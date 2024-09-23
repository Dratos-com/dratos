import lancedb
import os
from typing import List, Dict, Any

class LanceDBStore:
    def __init__(self, db_path: str):
        self.db = lancedb.connect(db_path)
        self.posts_table = self.db.create_table("posts", schema={
            "id": "string",
            "title": "string",
            "content": "string",
            "author": "string",
            "timestamp": "string",
            "upvotes": "int",
            "downvotes": "int",
            "comments": "int",
            "branches": "int"
        }, mode="overwrite")

    def add_post(self, post: Dict[str, Any]):
        self.posts_table.add([post])

    def get_all_posts(self) -> List[Dict[str, Any]]:
        return self.posts_table.to_pandas().to_dict('records')

    # ... other methods ...