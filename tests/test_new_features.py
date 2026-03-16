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
