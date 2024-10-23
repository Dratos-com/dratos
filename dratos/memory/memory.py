
from mem0 import Memory as Mem0Memory


class Memory(Mem0Memory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



# m = Memory()

# # For a user
# result = m.add("I like to take long walks on weekends.", user_id="alice", metadata={"category": "hobbies"})

# related_memories = m.search(query="Help me plan my weekend.", user_id="alice")

# # Get all memories
# all_memories = m.get_all(user_id="alice")
