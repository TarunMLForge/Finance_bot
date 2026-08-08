"""
Atlas AI Financial Assistant - Async MongoDB Database Manager
"""

from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
import pymongo
from app.core.config import settings
from app.core.logger import logger


class DatabaseManager:
    """Manages Async MongoDB connection lifecycle and indexes."""
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None
    is_connected: bool = False

    @classmethod
    async def connect(cls) -> None:
        """Connect to MongoDB and ensure collections and indexes exist."""
        try:
            logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI}...")
            client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=2000
            )
            # Ping database to verify connection
            await client.admin.command('ping')
            cls.client = client
            cls.db = cls.client[settings.MONGODB_DB_NAME]
            cls.is_connected = True
            logger.info(f"Successfully connected to MongoDB database '{settings.MONGODB_DB_NAME}'.")
            
            # Ensure indexes
            await cls.create_indexes()
        except Exception as e:
            cls.client = None
            cls.db = None
            cls.is_connected = False
            logger.warning(
                f"MongoDB connection attempt encountered an issue: {e}. "
                f"Atlas will operate with in-memory resilient storage."
            )

    @classmethod
    async def disconnect(cls) -> None:
        """Close MongoDB connection."""
        if cls.client is not None:
            logger.info("Closing MongoDB connection...")
            cls.client.close()
            cls.client = None
            cls.db = None
            logger.info("MongoDB connection closed.")

    @classmethod
    async def create_indexes(cls) -> None:
        """Creates indexes for users, conversations, and documents collections."""
        if cls.db is None:
            return
        
        try:
            # Users collection indexes
            users_col: AsyncIOMotorCollection = cls.db["users"]
            await users_col.create_index([("telegram_id", pymongo.ASCENDING)], unique=True)
            
            # Conversations collection indexes (for fast user lookup & recent message retrieval)
            conv_col: AsyncIOMotorCollection = cls.db["conversations"]
            await conv_col.create_index([
                ("telegram_id", pymongo.ASCENDING),
                ("timestamp", pymongo.DESCENDING)
            ])
            
            # Documents collection indexes
            doc_col: AsyncIOMotorCollection = cls.db["documents"]
            await doc_col.create_index([
                ("telegram_id", pymongo.ASCENDING),
                ("document_id", pymongo.ASCENDING)
            ], unique=True)
            
            logger.info("MongoDB database indexes successfully created and verified.")
        except Exception as e:
            logger.error(f"Error creating database indexes: {e}")

    @classmethod
    def get_collection(cls, collection_name: str) -> Optional[AsyncIOMotorCollection]:
        """Returns the requested collection if connected."""
        if cls.db is None:
            return None
        return cls.db[collection_name]


def get_users_collection() -> Optional[AsyncIOMotorCollection]:
    return DatabaseManager.get_collection("users")


def get_conversations_collection() -> Optional[AsyncIOMotorCollection]:
    return DatabaseManager.get_collection("conversations")


def get_documents_collection() -> Optional[AsyncIOMotorCollection]:
    return DatabaseManager.get_collection("documents")
