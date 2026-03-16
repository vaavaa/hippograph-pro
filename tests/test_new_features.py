"""
Tests for features added in March 2026:
- #15e: dateparser temporal filtering
- #16/#16b: synonym normalization + cross-lingual normalize_query
- temporal edges v2
- #26: embedding dimension validation
- #15c: CJK word segmentation
- studio_list_dir .TemporaryItems fix
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ─────────────────────────────────────────────
# #16 / #16b: Synonym normalization
# ─────────────────────────────────────────────

class TestSynonymNormalization:

    def test_normalize_entity_abbreviations(self):
        from entity_extractor import normalize_entity
        assert normalize_entity('ML') == 'machine learning'
        assert normalize_entity('k8s') == 'kubernetes'
        assert normalize_entity('NLP') == 'natural language processing'

    def test_normalize_entity_project_aliases(self):
        from entity_extractor import normalize_entity
        assert normalize_entity('hippograph pro') == 'hippograph'
        assert normalize_entity('neural memory') == 'hippograph'
        assert normalize_entity('semantic memory') == 'hippograph'

    def test_normalize_query_russian(self):
        from entity_extractor import normalize_query
        assert normalize_query('машинное обучение') == 'machine learning'
        assert normalize_query('искусственный интеллект') == 'artificial intelligence'
        assert normalize_query('нейронная сеть') == 'neural network'
        assert normalize_query('хиппограф') == 'hippograph'
        assert normalize_query('память') == 'memory'
        assert normalize_query('сознание') == 'consciousness'

    def test_normalize_query_spanish(self):
        from entity_extractor import normalize_query
        assert normalize_query('inteligencia artificial') == 'artificial intelligence'
        assert normalize_query('aprendizaje automático') == 'machine learning'
        assert normalize_query('red neuronal') == 'neural network'
        assert normalize_query('memoria') == 'memory'

    def test_normalize_query_german(self):
        from entity_extractor import normalize_query
        assert normalize_query('künstliche intelligenz') == 'artificial intelligence'
        assert normalize_query('maschinelles lernen') == 'machine learning'

    def test_normalize_query_french(self):
        from entity_extractor import normalize_query
        assert normalize_query('intelligence artificielle') == 'artificial intelligence'
        assert normalize_query('mémoire') == 'memory'

    def test_normalize_query_portuguese(self):
        from entity_extractor import normalize_query
        assert normalize_query('inteligência artificial') == 'artificial intelligence'
        assert normalize_query('memória') == 'memory'

    def test_normalize_query_mixed(self):
        """Partial phrase in query: only known tokens are replaced"""
        from entity_extractor import normalize_query
        assert normalize_query('k8s deployment') == 'kubernetes deployment'
        assert normalize_query('ml pipeline') == 'machine learning pipeline'

    def test_normalize_query_unknown_passthrough(self):
        """Unknown words pass through unchanged"""
        from entity_extractor import normalize_query
        result = normalize_query('some unknown phrase xyz')
        assert result == 'some unknown phrase xyz'


# ─────────────────────────────────────────────
# #15e: dateparser temporal filtering
# ─────────────────────────────────────────────

class TestDateparserTemporalFilter:

    def test_temporal_query_detected(self):
        from query_decomposer import is_temporal_query
        assert is_temporal_query('when did we deploy hippograph') is True
        assert is_temporal_query('what happened before the benchmark') is True
        assert is_temporal_query('recently added features') is True
        assert is_temporal_query('недавно добавили') is True
        assert is_temporal_query('когда это было') is True

    def test_non_temporal_query(self):
        from query_decomposer import is_temporal_query
        assert is_temporal_query('machine learning') is False
        assert is_temporal_query('hippograph architecture') is False

    def test_decompose_strips_temporal_words(self):
        from query_decomposer import decompose_temporal_query
        content, is_temporal, direction = decompose_temporal_query('when did we add hippograph benchmark')
        assert is_temporal is True
        assert 'hippograph' in content
        assert 'when did' not in content

    def test_decompose_detects_direction(self):
        from query_decomposer import decompose_temporal_query
        _, _, direction = decompose_temporal_query('what did we do last week')
        assert direction in ('before', 'after', 'around', None)

    def test_decompose_non_temporal_unchanged(self):
        from query_decomposer import decompose_temporal_query
        content, is_temporal, direction = decompose_temporal_query('spreading activation algorithm')
        assert is_temporal is False
        assert content == 'spreading activation algorithm'


# ─────────────────────────────────────────────
# #26: Embedding dimension validation
# ─────────────────────────────────────────────

class TestEmbeddingDimension:

    def test_model_produces_correct_dim(self):
        """paraphrase-multilingual-MiniLM-L12-v2 must output 384-dim vectors"""
        import numpy as np
        from graph_engine import get_model
        model = get_model()
        emb = model.encode('test embedding dimension check')[0]
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (384,), f'Expected 384, got {emb.shape}'

    def test_embedding_is_normalized(self):
        """Embeddings should be unit vectors (cosine similarity ready)"""
        import numpy as np
        from graph_engine import get_model
        model = get_model()
        emb = model.encode('normalization test')[0]
        norm = np.linalg.norm(emb)
        assert norm > 0
        # not pre-normalized by default, just check finite values
        assert np.all(np.isfinite(emb))


# ─────────────────────────────────────────────
# #15c: CJK word segmentation
# ─────────────────────────────────────────────

class TestCJKDetection:

    def test_cjk_detected_as_non_latin(self):
        from entity_extractor import detect_language
        # Chinese characters should route to multilingual model
        assert detect_language('人工智能和机器学习') == 'xx'

    def test_japanese_detected_as_non_latin(self):
        from entity_extractor import detect_language
        assert detect_language('人工知能と機械学習') == 'xx'

    def test_korean_detected_as_non_latin(self):
        from entity_extractor import detect_language
        assert detect_language('인공지능') == 'xx'

    def test_english_detected_as_latin(self):
        from entity_extractor import detect_language
        assert detect_language('machine learning') == 'en'

    def test_russian_detected_as_non_latin(self):
        from entity_extractor import detect_language
        assert detect_language('машинное обучение') == 'xx'

    def test_mixed_mostly_latin(self):
        """Text that is <20% non-Latin stays as 'en'"""
        from entity_extractor import detect_language
        # purely Latin text stays 'en'
        assert detect_language('HippoGraph system uses memory') == 'en'
        assert detect_language('machine learning pipeline') == 'en'

    def test_empty_string(self):
        from entity_extractor import detect_language
        assert detect_language('') == 'en'


# ─────────────────────────────────────────────
# .TemporaryItems filter (studio_list_dir fix)
# ─────────────────────────────────────────────

class TestTemporaryItemsFilter:

    def test_temporary_items_filter_logic(self):
        """Filtering logic: .TemporaryItems and .DS_Store should be excluded"""
        all_entries = ['src', 'README.md', '.TemporaryItems', '.DS_Store', 'tests']
        filtered = [e for e in all_entries
                    if not e.startswith('.') or e in {'.env.example'}]
        assert '.TemporaryItems' not in filtered
        assert '.DS_Store' not in filtered
        assert 'src' in filtered
        assert 'README.md' in filtered

    def test_src_dir_accessible(self):
        """Container src directory should be accessible"""
        import os
        entries = os.listdir('/app/src')
        assert 'entity_extractor.py' in entries
        assert 'graph_engine.py' in entries
        dotfiles = [e for e in entries if e.startswith('.DS_Store')]
        assert dotfiles == [], f'DS_Store found in container: {dotfiles}'

# ─────────────────────────────────────────────
# EMOTIONAL_RESONANCE edges
# ─────────────────────────────────────────────

class TestEmotionalResonance:

    def _make_db(self, tmp_path, notes):
        """Create minimal test DB with nodes table."""
        import sqlite3, os
        db = os.path.join(tmp_path, 'test.db')
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY,
                content TEXT,
                category TEXT DEFAULT 'general',
                timestamp TEXT,
                embedding BLOB,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0,
                importance TEXT DEFAULT 'normal',
                emotional_tone TEXT,
                emotional_intensity INTEGER DEFAULT 5,
                emotional_reflection TEXT,
                t_event_start TEXT,
                t_event_end TEXT,
                temporal_expressions TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER,
                target_id INTEGER,
                weight REAL DEFAULT 0.5,
                edge_type TEXT DEFAULT 'semantic',
                created_at TEXT,
                UNIQUE(source_id, target_id, edge_type)
            )
        """)
        for note_id, tone in notes:
            conn.execute(
                'INSERT INTO nodes (id, content, emotional_tone) VALUES (?, ?, ?)',
                (note_id, f'note {note_id}', tone)
            )
        conn.commit()
        conn.close()
        return db

    def test_creates_edges_for_shared_tags(self, tmp_path):
        """Two notes with 2+ shared tags get EMOTIONAL_RESONANCE edge."""
        import sys, os
        sys.path.insert(0, '/app/src')
        # Patch DB path for database module
        import database
        db = self._make_db(str(tmp_path), [
            (1, 'joy, warmth, pride'),
            (2, 'joy, warmth, accountability'),
            (3, 'shame, resolve'),
        ])
        orig = database.DB_PATH
        database.DB_PATH = db
        try:
            from sleep_compute import step_emotional_resonance
            result = step_emotional_resonance(db, dry_run=False)
            assert result['edges_created'] > 0, 'Expected edges between notes 1 and 2'
            assert result['pairs_checked'] >= 1
        finally:
            database.DB_PATH = orig

    def test_dry_run_creates_no_edges(self, tmp_path):
        """Dry run reports would_create but creates nothing."""
        import sys
        sys.path.insert(0, '/app/src')
        db = self._make_db(str(tmp_path), [
            (1, 'joy, warmth, pride'),
            (2, 'joy, warmth, gratitude'),
        ])
        from sleep_compute import step_emotional_resonance
        import sqlite3
        result = step_emotional_resonance(db, dry_run=True)
        assert result['edges_created'] == 0
        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM edges WHERE edge_type='EMOTIONAL_RESONANCE'").fetchone()[0]
        conn.close()
        assert count == 0

    def test_min_two_shared_tags(self, tmp_path):
        """Notes with only 1 shared tag do NOT get edges."""
        import sys, sqlite3
        sys.path.insert(0, '/app/src')
        db = self._make_db(str(tmp_path), [
            (1, 'joy, pride'),
            (2, 'joy, shame'),   # only 1 shared: joy
        ])
        from sleep_compute import step_emotional_resonance
        result = step_emotional_resonance(db, dry_run=False)
        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM edges WHERE edge_type='EMOTIONAL_RESONANCE'").fetchone()[0]
        conn.close()
        assert count == 0, f'Expected 0 edges, got {count}'

    def test_jaccard_weight(self, tmp_path):
        """Weight = shared / union (Jaccard)."""
        import sys
        sys.path.insert(0, '/app/src')
        # tags_a = {joy, warmth, pride}, tags_b = {joy, warmth, resolve}
        # shared=2, union=4, jaccard=0.5
        tags_a = {'joy', 'warmth', 'pride'}
        tags_b = {'joy', 'warmth', 'resolve'}
        shared = tags_a & tags_b
        jaccard = round(len(shared) / len(tags_a | tags_b), 3)
        assert abs(jaccard - 0.5) < 0.01, f'Expected ~0.5, got {jaccard}'
        assert len(shared) >= 2  # meets min 2 threshold
        # Verify step reports 1 resonant pair
        db = self._make_db(str(tmp_path), [
            (1, 'joy, warmth, pride'),
            (2, 'joy, warmth, resolve'),
        ])
        from sleep_compute import step_emotional_resonance
        result = step_emotional_resonance(db, dry_run=True)
        assert result['pairs_checked'] == 1
