import os
import sqlite3
from abc import ABC
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv

import log

load_dotenv()

DB_FILE = os.getenv("DB_FILE", "ai_host.db")

class DataModel(ABC):
    """Abstract base class for all data models.

    Subclasses must define `table_name` and a `schema` (dict of column -> type).
    `id` is always the default primary key.
    """

    table_name: str = ""
    schema: Dict[str, str] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.table_name:
            raise ValueError(f"{cls.__name__} must define a `table_name`.")
        if not cls.schema:
            raise ValueError(f"{cls.__name__} must define a `schema`.")
        # Ensure id is always present
        if "id" not in cls.schema:
            cls.schema = {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", **cls.schema}

    # ------------------------------------------------------------------ #
    # Connection helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _connect() -> sqlite3.Connection:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        return conn

    @classmethod
    def _ensure_table(cls) -> None:
        """Create the DB file and table if they don't exist."""
        cols = ", ".join(f"{name} {ctype}" for name, ctype in cls.schema.items())
        with cls._connect() as conn:
            conn.execute(f"CREATE TABLE IF NOT EXISTS {cls.table_name} ({cols})")
            conn.commit()
        log.debug(f"Table '{cls.table_name}' ready.")

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    @classmethod
    def create(cls, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[int]:
        """Create one or many elements. Returns list of inserted ids."""
        cls._ensure_table()
        if isinstance(data, dict):
            data = [data]

        inserted_ids: List[int] = []
        with cls._connect() as conn:
            for element in data:
                cols = ", ".join(element.keys())
                placeholders = ", ".join(["?"] * len(element))
                sql = f"INSERT INTO {cls.table_name} ({cols}) VALUES ({placeholders})"
                cur = conn.execute(sql, tuple(element.values()))
                inserted_ids.append(cur.lastrowid)
            conn.commit()
        log.debug(f"Inserted {len(inserted_ids)} row(s) into '{cls.table_name}'.")
        return inserted_ids

    @classmethod
    def read(
        cls,
        filters: Optional[Dict[str, Any]] = None,
        many: bool = True,
    ) -> Union[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Read one or many elements with an optional filter."""
        cls._ensure_table()
        query = f"SELECT * FROM {cls.table_name}"
        params: tuple = ()
        if filters:
            clauses = " AND ".join(f"{k} = ?" for k in filters)
            query += f" WHERE {clauses}"
            params = tuple(filters.values())

        with cls._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        results = [dict(row) for row in rows]
        if many:
            return results
        return results[0] if results else None

    @classmethod
    def update(
        cls,
        filters: Dict[str, Any],
        updates: Dict[str, Any],
    ) -> int:
        """Update one or many elements matching `filters`. Returns affected rows."""
        cls._ensure_table()
        if not filters:
            raise ValueError("`filters` cannot be empty for update().")

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        where_clause = " AND ".join(f"{k} = ?" for k in filters)
        params = tuple(updates.values()) + tuple(filters.values())
        sql = f"UPDATE {cls.table_name} SET {set_clause} WHERE {where_clause}"

        with cls._connect() as conn:
            cur = conn.execute(sql, params)
            conn.commit()
            affected = cur.rowcount
        log.debug(f"Updated {affected} row(s) in '{cls.table_name}'.")
        return affected

    @classmethod
    def delete(cls, filters: Dict[str, Any]) -> int:
        """Delete one or many elements matching `filters`. Returns affected rows."""
        cls._ensure_table()
        if not filters:
            raise ValueError("`filters` cannot be empty for delete().")

        where_clause = " AND ".join(f"{k} = ?" for k in filters)
        sql = f"DELETE FROM {cls.table_name} WHERE {where_clause}"

        with cls._connect() as conn:
            cur = conn.execute(sql, tuple(filters.values()))
            conn.commit()
            affected = cur.rowcount
        log.debug(f"Deleted {affected} row(s) from '{cls.table_name}'.")
        return affected
