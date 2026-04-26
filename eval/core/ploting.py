import json
import numpy as np
import matplotlib.pyplot as plt
from math import pi

with open('eval/results/eval_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

score_dict = {item['scenario_id']: item['scores']['total'] for item in data}

pillars = ['Temporal', 'Statistical', 'System\nFeedback', 'Regime', 'Risk']
mapping = {
    'S001': 0, 'S002': 0, 'S005': 0,  # Temporal
    'S003': 1, 'S009': 1, 'S011': 1,  # Statistical
    'S004': 2, 'S006': 2,              # System Feedback
    'S007': 3, 'S008': 3,              # Regime
    'S010': 4, 'S012': 4               # Risk
}

pillar_scores = []
for i in range(5):
    scens = [s for s, p in mapping.items() if p == i]
    vals = [score_dict.get(s, 0) for s in scens]
    pillar_scores.append(np.average(vals) if vals else 0)

N = 5
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
pillar_scores += pillar_scores[:1]

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

ax.plot(angles, pillar_scores, 'b-', lw=2.5, label='Causal Backbone (avg total)')
ax.fill(angles, pillar_scores, 'b', alpha=0.25)
ax.plot(angles, pillar_scores, 'orange', lw=2.5, label='Utility Aura (avg total)')
ax.fill(angles, pillar_scores, 'orange', alpha=0.12)

ax.set_thetagrids([a*180/pi for a in angles[:-1]], pillars, fontsize=11)
ax.set_ylim(0, 120)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title('FinDecision-Eval 5-Pillar Radar\n(avg total per pillar)')

status = "FAIL" if np.average(pillar_scores[:-1]) <= 0 else "CAUTION" if np.average(pillar_scores[:-1]) < 60 else "PASS"
print("GLOBAL STATUS:", status)
plt.show()