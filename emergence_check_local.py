import sqlite3, math, random

DB = '/Volumes/Balances/hippograph-pro/data/memory.db'
conn = sqlite3.connect(DB)

# === PHI PROXY ===
total_nodes = conn.execute('SELECT COUNT(*) FROM nodes').fetchone()[0]
communities = {}
rows = conn.execute('SELECT id FROM nodes').fetchall()
for (nid,) in rows:
    communities[nid] = nid % 10

cross = conn.execute("""
    SELECT COUNT(*) FROM edges e
    JOIN nodes n1 ON e.source_id = n1.id
    JOIN nodes n2 ON e.target_id = n2.id
    WHERE e.weight > 0.3
""").fetchone()[0]

phi_raw = cross / max(total_nodes, 1)
phi_norm = min(1.0, phi_raw / 100)
print(f'phi_proxy: {phi_norm:.4f} (raw={phi_raw:.2f}, cross_edges={cross}, nodes={total_nodes})')

# === SELF-REFERENTIAL P@5 ===
queries = [
    ('identity continuity substrate memory', ['self-identity', 'consciousness-research']),
    ('HippoGraph spreading activation graph', ['development', 'milestone', 'breakthrough']),
    ('Claudé name persona learned skill', ['self-identity', 'learned-skill', 'self-reflection']),
]
hits = 0
for q, target_cats in queries:
    words = q.lower().split()
    results = conn.execute("""
        SELECT n.id, n.category, n.content FROM nodes n
        WHERE " " || lower(n.content) || " " LIKE ?
        LIMIT 5
    """, (f'%{words[0]}%',)).fetchall()
    found = any(r[1] in target_cats for r in results)
    if found:
        hits += 1
self_ref = hits / len(queries)
print(f'self_ref P@5: {self_ref:.4f} ({hits}/{len(queries)} queries hit target categories)')

# === CONVERGENCE ===
seeds = [random.randint(1, total_nodes) for _ in range(5)]
activations = []
for seed in seeds:
    neighbors = conn.execute("""
        SELECT target_id, weight FROM edges WHERE source_id=? AND weight>0.3 LIMIT 10
    """, (seed,)).fetchall()
    act = sum(w for _, w in neighbors) / max(len(neighbors), 1)
    activations.append(act)
if len(activations) > 1:
    mean = sum(activations) / len(activations)
    variance = sum((a - mean)**2 for a in activations) / len(activations)
    convergence = 1.0 / (1.0 + math.sqrt(variance))
else:
    convergence = 0.0
print(f'convergence: {convergence:.4f}')

# === EDGE DIVERSITY ===
edge_types = conn.execute('SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type').fetchall()
total_edges = sum(c for _, c in edge_types)
entity_edges = next((c for t, c in edge_types if t == 'entity'), 0)
diversity = 1.0 - (entity_edges / max(total_edges, 1))
print(f'edge_diversity: {diversity:.4f} (entity={entity_edges}/{total_edges})')

# === COMPOSITE ===
composite = (phi_norm + self_ref + convergence + diversity) / 4
print(f'\nEMERGENCE SCORE: {composite:.4f}')
print(f'  phi_norm={phi_norm:.3f} self_ref={self_ref:.3f} convergence={convergence:.3f} diversity={diversity:.3f}')

conn.close()