import sqlite3
conn = sqlite3.connect('/Volumes/Balances/hippograph-pro/data/memory.db')
total = conn.execute('SELECT COUNT(*) FROM emergence_log').fetchone()[0]
rows = conn.execute('SELECT id, timestamp, phi_proxy, self_ref_precision, convergence_score, composite_score FROM emergence_log ORDER BY id DESC LIMIT 10').fetchall()
print(f'Total measurements: {total}')
print()
print('Last 10:')
for r in rows:
    print(f'  #{r[0]} {r[1][:19]} | phi={r[2]:.3f} self_ref={r[3]:.3f} conv={r[4]:.4f} composite={r[5]:.4f}')
print()
first = conn.execute('SELECT composite_score FROM emergence_log ORDER BY id ASC LIMIT 1').fetchone()[0]
last = rows[0][5]
print(f'First ever: {first:.4f}')
print(f'Latest:     {last:.4f}')
print(f'Delta:      +{last-first:.4f} ({(last-first)/first*100:.1f}%)')
conn.close()