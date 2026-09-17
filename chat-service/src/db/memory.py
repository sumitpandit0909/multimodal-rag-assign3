from motor.motor_asyncio import AsyncIOMotorDatabase
from typing import List, Dict
import datetime

class MongoChatMemory:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db["conversations"]

    async def get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        cursor = self.collection.find({"session_id": session_id}).sort("timestamp", 1).limit(limit)
        return await cursor.to_list(length=limit)

    async def add_turn(self, session_id: str, user_message: str, assistant_message: str, sources: List[Dict]):
        await self.collection.insert_one({
            "session_id": session_id,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "sources": sources,
            "timestamp": datetime.datetime.utcnow()
        })
