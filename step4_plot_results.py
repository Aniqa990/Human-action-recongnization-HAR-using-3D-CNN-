"""
STEP 4: plot_results.py
Reads train.txt / val.txt / train_batch.txt and generates training graphs.

Usage:
    python step4_plot_results.py \
        --result_path "G:/My Drive/Segmented_FYP_DATA_jpg_raw/results_slowfast"
"""

import argparse, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str,
                    default='G:/My Drive/Segmented_FYP_DATA_jpg_raw/results_slowfast')
args = parser.parse_args()
rp = args.result_path

def find(folder, names):
    for n in names:
        p = os.path.join(folder, n)
        if os.path.exists(p): return p
    return None

def parse_epoch_log(filepath):
    epochs, losses, accs = [], [], []
    if not filepath: return epochs, losses, accs
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = re.split(r'[\t,\s]+', line)
            try: float(parts[0])
            except: continue
            if len(parts) >= 3:
                epochs.append(int(float(parts[0])))
                losses.append(float(parts[1]))
                accs.append(float(parts[2]))
    return epochs, losses, accs

def parse_batch_log(filepath):
    iters, losses, accs = [], [], []
    if not filepath: return iters, losses, accs
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = re.split(r'[\t,\s]+', line)
            try: float(parts[0])
            except: continue
            if len(parts) >= 5:
                iters.append(int(float(parts[2])))
                losses.append(float(parts[3]))
                accs.append(float(parts[4]))
    return iters, losses, accs

train_ep, train_lo, train_ac = parse_epoch_log(find(rp, ['train.txt','train.log']))
val_ep,   val_lo,   val_ac   = parse_epoch_log(find(rp, ['val.txt',  'val.log']))
b_it,     b_lo,     b_ac     = parse_batch_log(find(rp, ['train_batch.txt','train_batch.log']))

has_train = len(train_ep) > 0
has_val   = len(val_ep)   > 0
has_batch = len(b_it)     > 0

print(f"Train epochs: {train_ep}")
print(f"Val   epochs: {val_ep}")
print(f"Batch steps : {len(b_it)}")

if not has_train and not has_val:
    print("❌ No data found"); exit(1)

try:    plt.style.use('seaborn-v0_8-darkgrid')
except: plt.style.use('ggplot')

C = dict(tl='#E74C3C', vl='#3498DB', ta='#2ECC71', va='#F39C12', b='#9B59B6')
fig = plt.figure(figsize=(18, 13))
fig.suptitle('SlowFast / Two-Stream  |  FYP Training Results\n(Child Safety Action Recognition)',
             fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

# 1. Loss
ax1 = fig.add_subplot(gs[0,0])
if has_train: ax1.plot(train_ep, train_lo, color=C['tl'], marker='o', markersize=5, linewidth=2, label='Train Loss')
if has_val:   ax1.plot(val_ep,   val_lo,   color=C['vl'], marker='s', markersize=5, linewidth=2, linestyle='--', label='Val Loss')
ax1.set_title('Training & Validation Loss', fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()

# 2. Accuracy
ax2 = fig.add_subplot(gs[0,1])
if has_train: ax2.plot(train_ep, [a*100 for a in train_ac], color=C['ta'], marker='o', markersize=5, linewidth=2, label='Train Acc')
if has_val:   ax2.plot(val_ep,   [a*100 for a in val_ac],   color=C['va'], marker='s', markersize=5, linewidth=2, linestyle='--', label='Val Acc')
ax2.set_title('Training & Validation Accuracy', fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.legend()

# 3. Batch loss
ax3 = fig.add_subplot(gs[0,2])
if has_batch:
    w = max(1, len(b_lo)//60)
    sm = np.convolve(b_lo, np.ones(w)/w, mode='valid')
    ax3.plot(b_it, b_lo, color=C['b'], alpha=0.15, linewidth=0.7)
    ax3.plot(b_it[w-1:], sm, color=C['b'], linewidth=2, label='Smoothed')
    ax3.set_title('Batch-Level Loss', fontweight='bold')
    ax3.set_xlabel('Iteration'); ax3.set_ylabel('Loss'); ax3.legend()
else:
    ax3.text(0.5,0.5,'No batch log',ha='center',va='center',transform=ax3.transAxes,fontsize=12)
    ax3.set_title('Batch-Level Loss', fontweight='bold')

# 4. Batch accuracy
ax4 = fig.add_subplot(gs[1,0])
if has_batch:
    w2 = max(1, len(b_ac)//60)
    sm2 = np.convolve(b_ac, np.ones(w2)/w2, mode='valid')
    ax4.plot(b_it, [a*100 for a in b_ac], color=C['ta'], alpha=0.15, linewidth=0.7)
    ax4.plot(b_it[w2-1:], [a*100 for a in sm2], color=C['ta'], linewidth=2, label='Smoothed')
    ax4.set_title('Batch-Level Accuracy', fontweight='bold')
    ax4.set_xlabel('Iteration'); ax4.set_ylabel('Accuracy (%)'); ax4.legend()
else:
    ax4.text(0.5,0.5,'No batch log',ha='center',va='center',transform=ax4.transAxes,fontsize=12)
    ax4.set_title('Batch-Level Accuracy', fontweight='bold')

# 5. Summary table
ax5 = fig.add_subplot(gs[1,1]); ax5.axis('off')
rows = []
if has_train:
    bta=max(train_ac); bte=train_ep[train_ac.index(bta)]
    rows += [('Best Train Acc', f'{bta*100:.2f}%  (ep {bte})'),
             ('Min Train Loss', f'{min(train_lo):.4f}'),
             ('Final Train Acc',f'{train_ac[-1]*100:.2f}%')]
if has_val:
    bva=max(val_ac); bve=val_ep[val_ac.index(bva)]
    rows += [('Best Val Acc', f'{bva*100:.2f}%  (ep {bve})'),
             ('Min Val Loss', f'{min(val_lo):.4f}'),
             ('Final Val Acc',f'{val_ac[-1]*100:.2f}%')]
if has_train:
    rows += [('Epochs', f'{min(train_ep)} → {max(train_ep)}')]
if rows:
    tbl = ax5.table(cellText=rows, colLabels=['Metric','Value'],
                    cellLoc='center', loc='center', bbox=[0,0.05,1,0.90])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (r,c), cell in tbl.get_celld().items():
        if r==0: cell.set_facecolor('#2C3E50'); cell.set_text_props(color='white',fontweight='bold')
        elif r%2==0: cell.set_facecolor('#ECF0F1')
ax5.set_title('Summary Statistics', fontweight='bold', pad=10)

# 6. Combined dual axis
ax6 = fig.add_subplot(gs[1,2])
ax6b = ax6.twinx()
if has_train:
    ax6.plot(train_ep, train_lo, color=C['tl'], marker='o', markersize=4, linewidth=2, label='Train Loss')
    ax6b.plot(train_ep, [a*100 for a in train_ac], color=C['ta'], marker='^', markersize=4, linewidth=2, linestyle=':', label='Train Acc%')
if has_val:
    ax6.plot(val_ep, val_lo, color=C['vl'], marker='s', markersize=4, linewidth=2, linestyle='--', label='Val Loss')
    ax6b.plot(val_ep, [a*100 for a in val_ac], color=C['va'], marker='v', markersize=4, linewidth=2, linestyle='-.', label='Val Acc%')
ax6.set_xlabel('Epoch'); ax6.set_ylabel('Loss', color=C['tl'])
ax6b.set_ylabel('Accuracy (%)', color=C['ta'])
ax6.set_title('Loss & Accuracy Combined', fontweight='bold')
all_lines = ax6.get_lines() + ax6b.get_lines()
ax6.legend(all_lines, [l.get_label() for l in all_lines], fontsize=8)

out = os.path.join(rp, 'training_graphs.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {out}")
