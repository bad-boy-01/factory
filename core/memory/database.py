"""
Memory Database — Novel Video Factory v5
SQLAlchemy-backed knowledge store for story entities.

Tables:
  Character     — visual DNA + canonical name
  Location      — visual tags + background image path
  Relationship  — character interactions
  WorldConcept  — lore, items, power systems
"""
import json
import logging
import os
import re
from contextlib import contextmanager
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import Column, Integer, JSON, String, Text, create_engine
    from sqlalchemy.orm import DeclarativeBase, sessionmaker

    class Base(DeclarativeBase):
        pass

    class Character(Base):
        __tablename__ = "characters"
        id = Column(String, primary_key=True)
        canonical_name = Column(String, nullable=False, unique=True)
        visual_dna = Column(JSON, default=dict)
        current_state = Column(JSON, default=dict)

    class Location(Base):
        __tablename__ = "locations"
        id = Column(Integer, primary_key=True, autoincrement=True)
        canonical_name = Column(String, nullable=False, unique=True)
        description = Column(Text, default="")
        visual_tags = Column(Text, default="")
        background_path = Column(String, default="")

    class Relationship(Base):
        __tablename__ = "relationships"
        id = Column(Integer, primary_key=True, autoincrement=True)
        character_a = Column(String)
        character_b = Column(String)
        relationship_type = Column(String, default="other")
        description = Column(Text, default="")

    class WorldConcept(Base):
        __tablename__ = "world_concepts"
        id = Column(Integer, primary_key=True, autoincrement=True)
        concept_type = Column(String, default="misc")
        name = Column(String)
        description = Column(Text, default="")

    class Event(Base):
        __tablename__ = "events"
        id = Column(Integer, primary_key=True, autoincrement=True)
        summary = Column(Text, nullable=False)
        importance = Column(Integer, default=5)
        involved_characters = Column(Text, default="")
        location = Column(String, default="")
        source_chunk = Column(String, default="")

    SQLALCHEMY_AVAILABLE = True

except ImportError:
    logger.warning("SQLAlchemy not installed — using in-memory dict fallback")
    SQLALCHEMY_AVAILABLE = False


class MemoryEngine:
    """
    Central knowledge store for the novel's characters, locations, and lore.
    Falls back to an in-memory dict if SQLAlchemy is not installed.
    """
    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        mem_dir = os.path.join(project_dir, "memory")
        os.makedirs(mem_dir, exist_ok=True)

        self._in_memory = not SQLALCHEMY_AVAILABLE
        if self._in_memory:
            self._chars: Dict[str, dict] = {}
            self._locs: Dict[str, dict] = {}
            self._rels: List[dict] = []
            self._concepts: List[dict] = []
            self._events: List[dict] = []
            return

        db_path = os.path.join(mem_dir, "novel_memory.db")
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(engine)
        self.SessionLocal = sessionmaker(bind=engine)

    @contextmanager
    def Session(self):
        if self._in_memory:
            yield None
            return
        s = self.SessionLocal()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ── Events ───────────────────────────────────────────────────────────────
    def add_event(self, summary: str, importance: int = 5, involved_characters: str = "",
                  location: str = "", source_chunk: str = ""):
        if self._in_memory:
            self._events.append({
                "summary": summary, "importance": importance,
                "involved_characters": involved_characters, "location": location,
                "source_chunk": source_chunk
            })
            return

        with self.Session() as s:
            s.add(Event(summary=summary, importance=importance,
                        involved_characters=involved_characters, location=location,
                        source_chunk=source_chunk))

    def get_events_by_chunk(self, source_chunk: str) -> List[Dict]:
        if self._in_memory:
            return [e for e in self._events if e.get("source_chunk") == source_chunk]

        with self.Session() as s:
            events = s.query(Event).filter_by(source_chunk=source_chunk).all()
            return [{"summary": e.summary, "importance": e.importance,
                     "involved_characters": e.involved_characters,
                     "location": e.location, "source_chunk": e.source_chunk}
                    for e in events]

    def get_all_events(self) -> List[Dict]:
        if self._in_memory:
            return self._events

        with self.Session() as s:
            return [{"summary": e.summary, "importance": e.importance,
                     "involved_characters": e.involved_characters,
                     "location": e.location, "source_chunk": e.source_chunk}
                    for e in s.query(Event).all()]

    # ── Characters ───────────────────────────────────────────────────────────
    def add_character(self, char_id: str, name: str, visual_dna: dict, current_state: dict = None):
        if current_state is None:
            current_state = {}
            
        # Deduplication: strip descriptors like "(Old Woman)" or "(Mother)"
        # e.g., "Mrs. Xu Zhang (Mother)" -> "Mrs. Xu Zhang"
        base_name = re.sub(r'\s*\([^)]*\)', '', name).strip()
            
        if self._in_memory:
            # Check for existing alias
            found_key = None
            for key in self._chars:
                if key == name or key == base_name or key.startswith(base_name):
                    found_key = key
                    break
            
            if found_key:
                self._chars[found_key]["visual_dna"].update(visual_dna)
                self._chars[found_key]["current_state"].update(current_state)
            else:
                self._chars[name] = {"id": char_id, "canonical_name": name,
                                     "visual_dna": visual_dna, "current_state": current_state}
            return

        with self.Session() as s:
            # FIX: previously this only matched on exact (lowercased) full
            # name equality, so "Mrs. Xu Zhang" and "Mrs. Xu Zhang (Old
            # Woman)" were never recognized as the same character — they
            # ended up as two separate DB rows, each generating its own
            # character sheet / reference image for what should be one
            # person. Match the same way the in-memory fallback path
            # already does (base_name from either side matching the
            # other's full name), instead of just `== name.lower()`.
            name_lower = name.lower()
            base_lower = base_name.lower()
            existing = None
            for cand in s.query(Character).all():
                cand_lower = cand.canonical_name.lower()
                if cand_lower == name_lower or cand_lower == base_lower or cand_lower.startswith(base_lower):
                    existing = cand
                    break
            
            if existing:
                e_dna = existing.visual_dna if existing.visual_dna is not None else {}
                v_dna = visual_dna if visual_dna is not None else {}
                
                merged_dna = dict(e_dna)
                
                # Appearance accumulation
                old_conf = float(merged_dna.get("appearance_confidence", 0.0))
                new_conf = float(v_dna.get("appearance_confidence", 0.0))
                
                for k, v in v_dna.items():
                    if k == "appearance_confidence":
                        merged_dna[k] = max(old_conf, new_conf)
                        continue
                        
                    # If it's a list (like distinctive_features), append and deduplicate
                    if isinstance(v, list):
                        existing_list = merged_dna.get(k, [])
                        if not isinstance(existing_list, list):
                            existing_list = [existing_list] if existing_list else []
                        merged_list = existing_list + [x for x in v if x not in existing_list and x]
                        if merged_list:
                            merged_dna[k] = merged_list
                        continue
                        
                    # If it's a string, only overwrite if populated AND (it was empty OR new confidence > old)
                    if isinstance(v, str) and v.strip() and v.strip().lower() not in {"none", "unknown", "n/a", "not specified"}:
                        existing_val = merged_dna.get(k, "")
                        is_empty = not existing_val or existing_val.strip().lower() in {"none", "unknown", "n/a", "not specified"}
                        
                        if is_empty:
                            merged_dna[k] = v.strip()
                        elif new_conf > old_conf:
                            merged_dna[k] = v.strip()
                        elif new_conf == old_conf and v.strip() not in existing_val:
                            # Append to existing description if same confidence but different detail
                            merged_dna[k] = f"{existing_val}, {v.strip()}"

                existing.visual_dna = merged_dna
                
                if existing.current_state:
                    merged_state = {**existing.current_state, **current_state}
                else:
                    merged_state = current_state
                    
                # Importance is persistent, keep highest
                if "importance" in existing.current_state and "importance" in current_state:
                    merged_state["importance"] = max(existing.current_state["importance"], current_state["importance"])
                    
                existing.current_state = merged_state
            else:
                s.add(Character(id=char_id, canonical_name=name, visual_dna=visual_dna, current_state=current_state))

    def get_character_by_name(self, name: str) -> Optional[Dict]:
        if self._in_memory:
            return self._chars.get(name)

        with self.Session() as s:
            c = s.query(Character).filter_by(canonical_name=name).first()
            if c:
                return {"id": c.id, "canonical_name": c.canonical_name,
                        "visual_dna": c.visual_dna, "current_state": c.current_state}
        return None

    def get_all_characters(self) -> List[Dict]:
        if self._in_memory:
            return list(self._chars.values())

        with self.Session() as s:
            return [{"id": c.id, "canonical_name": c.canonical_name,
                     "visual_dna": c.visual_dna, "current_state": c.current_state}
                    for c in s.query(Character).all()]

    # ── Locations ─────────────────────────────────────────────────────────────
    def add_location(self, name: str, description: str = "", visual_tags: str = ""):
        if self._in_memory:
            if name not in self._locs:
                self._locs[name] = {"canonical_name": name, "description": description,
                                    "visual_tags": visual_tags, "background_path": ""}
            return

        with self.Session() as s:
            from sqlalchemy import func
            if not s.query(Location).filter(func.lower(Location.canonical_name) == name.lower()).first():
                s.add(Location(canonical_name=name, description=description,
                               visual_tags=visual_tags))

    def get_location_by_name(self, name: str) -> Optional[Dict]:
        if self._in_memory:
            return self._locs.get(name)

        with self.Session() as s:
            loc = s.query(Location).filter_by(canonical_name=name).first()
            if loc:
                return {"canonical_name": loc.canonical_name,
                        "description": loc.description,
                        "visual_tags": loc.visual_tags,
                        "background_path": loc.background_path}
        return None

    def get_all_locations(self) -> List[Dict]:
        if self._in_memory:
            return list(self._locs.values())

        with self.Session() as s:
            return [{"canonical_name": l.canonical_name, "description": l.description,
                     "visual_tags": l.visual_tags, "background_path": l.background_path}
                    for l in s.query(Location).all()]

    def update_location_background(self, name: str, path: str):
        if self._in_memory:
            if name in self._locs:
                self._locs[name]["background_path"] = path
            return

        with self.Session() as s:
            loc = s.query(Location).filter_by(canonical_name=name).first()
            if loc:
                loc.background_path = path

    # ── Relationships ─────────────────────────────────────────────────────────
    def add_relationship(self, char_a: str, char_b: str, rel_type: str = "other",
                         description: str = ""):
        char_1, char_2 = sorted([char_a, char_b])
        if self._in_memory:
            self._rels.append({"character_a": char_1, "character_b": char_2,
                                "type": rel_type, "description": description})
            return

        with self.Session() as s:
            existing = s.query(Relationship).filter(
                Relationship.character_a == char_1,
                Relationship.character_b == char_2,
            ).first()
            if not existing:
                s.add(Relationship(character_a=char_1, character_b=char_2,
                                   relationship_type=rel_type, description=description))

    def get_all_relationships(self) -> List[Dict]:
        if self._in_memory:
            return self._rels

        with self.Session() as s:
            return [{"character_a": r.character_a, "character_b": r.character_b,
                     "type": r.relationship_type, "description": r.description}
                    for r in s.query(Relationship).all()]

    def get_relationship_staging(self, character_names: List[str]) -> str:
        """Returns a short comma-separated staging tag for scene composition."""
        rels = self.get_all_relationships()
        tags = []
        for rel in rels:
            if rel["character_a"] in character_names and rel["character_b"] in character_names:
                rt = rel.get("type", "other")
                if rt == "rivals":
                    tags.append("confrontational stance, tension")
                elif rt in ("allies", "friends"):
                    tags.append("standing together, comradery")
                elif rt == "romance":
                    tags.append("close proximity, warm atmosphere")
        return ", ".join(tags[:2]) if tags else ""

    # ── World Concepts ────────────────────────────────────────────────────────
    def add_world_concept(self, concept_type: str, name: str, description: str = ""):
        if self._in_memory:
            self._concepts.append({"type": concept_type, "name": name,
                                   "description": description})
            return

        with self.Session() as s:
            if not s.query(WorldConcept).filter_by(name=name).first():
                s.add(WorldConcept(concept_type=concept_type, name=name,
                                   description=description))

    # ── Export ────────────────────────────────────────────────────────────────
    def export_to_json(self) -> Dict:
        return {
            "characters": self.get_all_characters(),
            "locations": self.get_all_locations(),
            "relationships": self.get_all_relationships(),
            "events": self.get_all_events(),
        }
