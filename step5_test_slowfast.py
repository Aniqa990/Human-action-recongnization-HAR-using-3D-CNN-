"""
STEP 5: test_slowfast.py
Runs inference on unseen test videos using trained SlowFast / Two-Stream model.
Generates confusion matrix, per-class metrics, confidence plots, CSV report.

Usage:
    python step5_test_slowfast.py \
        --jpg_root    "G:/My Drive/FYP_TEST_JPG" \
        --annotation  "G:/My Drive/FYP_TEST_JPG/test_annotation.json" \
        --checkpoint  "G:/My Drive/.../results_slowfast/best_model.pth" \
        --result_path "G:/My Drive/FYP_TEST_JPG/test_results_slowfast"
"""

import os, json, argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SK = True
except:
    HAS_SK = False
    print("[WARN] pip install scikit-learn for full metrics")

parser = argparse.ArgumentParser()
parser.add_argument('--jpg_root',    type=str,
                    default='G:/My Drive/FYP_TEST_JPG')
parser.add_argument('--annotation',  type=str,
                    default='G:/My Drive/FYP_TEST_JPG/test_annotation.json')
parser.add_argument('--checkpoint',  type=str,
                    default='G:/My Drive/Segmented_FYP_DATA_jpg_raw/results_slowfast/best_model.pth')
parser.add_argument('--result_path', type=str,
                    default='G:/My Drive/FYP_TEST_JPG/test_results_slowfast')
parser.add_argument('--slow_frames', type=int, default=8)
parser.add_argument('--fast_frames', type=int, default=32)
parser.add_argument('--img_size',    type=int, default=112)
parser.add_argument('--batch_size',  type=int, default=4)
parser.add_argument('--n_workers',   type=int, default=2)
args = parser.parse_args()

CLASSES  = ['fight', 'Normal', 'unsafeClimb', 'unsafeJump', 'unsafeThrow']
C2I      = {c: i for i, c in enumerate(CLASSES)}
os.makedirs(args.result_path, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Dataset ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.45,0.45,0.45],[0.225,0.225,0.225]),
])

class TestDataset(Dataset):
    def __init__(self, jpg_root, annotation_path, slow_frames, fast_frames):
        self.jpg_root    = jpg_root
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        with open(annotation_path,'r') as f:
            ann = json.load(f)
        self.samples = []
        for vid, info in ann['database'].items():
            lbl = info['annotations']['label']
            idx = C2I.get(lbl,-1)
            if idx < 0: continue
            for cand in [os.path.join(jpg_root,lbl,vid), os.path.join(jpg_root,vid)]:
                if os.path.isdir(cand):
                    self.samples.append((cand,idx,lbl,vid)); break
        print(f"Test samples: {len(self.samples)}")

    def _load(self, vid_dir, n):
        files = sorted([f for f in os.listdir(vid_dir) if f.endswith('.jpg')])
        total = len(files)
        if total == 0: return torch.zeros(3,n,args.img_size,args.img_size)
        idx = np.linspace(0,total-1,n,dtype=int)
        frames = []
        for i in idx:
            try:    img = Image.open(os.path.join(vid_dir,files[i])).convert('RGB')
            except: img = Image.new('RGB',(args.img_size,args.img_size))
            frames.append(transform(img))
        return torch.stack(frames,0).permute(1,0,2,3)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        vid_dir,label,lbl_name,vid_name = self.samples[idx]
        return [self._load(vid_dir,self.slow_frames),
                self._load(vid_dir,self.fast_frames)], label, lbl_name, vid_name

def collate_fn(batch):
    slow   = torch.stack([b[0][0] for b in batch])
    fast   = torch.stack([b[0][1] for b in batch])
    labels = torch.tensor([b[1]   for b in batch], dtype=torch.long)
    lnames = [b[2] for b in batch]
    vnames = [b[3] for b in batch]
    return [slow,fast], labels, lnames, vnames

# ── Load model ─────────────────────────────────────────────────────────────────
def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_type = ckpt.get('model_type','two_stream')
    state = ckpt.get('state_dict', ckpt)

    if model_type in ('slowfast_hub','slowfast_ptv'):
        try:
            model = torch.hub.load('facebookresearch/pytorchvideo:main',
                                   'slowfast_r50', pretrained=False)
            in_feat = model.blocks[-1].proj.in_features
            model.blocks[-1].proj = nn.Linear(in_feat, len(CLASSES))
        except:
            model_type = 'two_stream'

    if model_type == 'two_stream':
        from torchvision.models.video import r3d_18
        class TwoStreamModel(nn.Module):
            def __init__(self,n):
                super().__init__()
                slow = r3d_18(weights=None); fast = r3d_18(weights=None)
                self.slow_encoder = nn.Sequential(*list(slow.children())[:-1])
                self.fast_encoder = nn.Sequential(*list(fast.children())[:-1])
                self.classifier   = nn.Sequential(
                    nn.Flatten(), nn.Dropout(0.5),
                    nn.Linear(512+512,512), nn.ReLU(inplace=True),
                    nn.Dropout(0.3), nn.Linear(512,n))
            def forward(self,inputs):
                sf = self.slow_encoder(inputs[0]).squeeze(-1).squeeze(-1).squeeze(-1)
                ff = self.fast_encoder(inputs[1]).squeeze(-1).squeeze(-1).squeeze(-1)
                return self.classifier(torch.cat([sf,ff],dim=1))
        model = TwoStreamModel(len(CLASSES))

    new_state = {k.replace('module.',''):v for k,v in state.items()}
    model.load_state_dict(new_state, strict=False)
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded ({model_type})")
    return model

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, loader):
    all_preds, all_labels, all_probs, all_names = [], [], [], []
    with torch.no_grad():
        for i,(inputs,labels,_,vnames) in enumerate(loader):
            inputs = [x.to(device) for x in inputs]
            out    = model(inputs)
            probs  = torch.softmax(out,dim=1)
            preds  = probs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_names.extend(vnames)
            print(f"  Batch {i+1}/{len(loader)}", end='\r')
    print()
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_names

# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm, path):
    fig,ax = plt.subplots(figsize=(8,7))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im,ax=ax)
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES,rotation=35,ha='right',fontsize=10)
    ax.set_yticklabels(CLASSES,fontsize=10)
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=12,
                    color='white' if cm[i,j]>thresh else 'black')
    ax.set_ylabel('True Label',fontsize=12); ax.set_xlabel('Predicted Label',fontsize=12)
    ax.set_title('Confusion Matrix — Unseen Test Data\n(SlowFast FYP)',fontsize=13,fontweight='bold')
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")

def plot_per_class_metrics(report, path):
    metrics = ['precision','recall','f1-score']
    x = np.arange(len(CLASSES)); w = 0.25
    colors = ['#3498DB','#2ECC71','#E74C3C']
    fig,ax = plt.subplots(figsize=(11,6))
    for i,(m,c) in enumerate(zip(metrics,colors)):
        vals = [report.get(cls,{}).get(m,0) for cls in CLASSES]
        ax.bar(x+i*w, vals, w, label=m.capitalize(), color=c, alpha=0.85)
    ax.set_xticks(x+w); ax.set_xticklabels(CLASSES,fontsize=11)
    ax.set_ylabel('Score',fontsize=12); ax.set_ylim(0,1.1)
    ax.axhline(0.8,color='gray',linestyle='--',linewidth=1,alpha=0.5,label='0.8 target')
    ax.set_title('Per-Class Precision / Recall / F1',fontsize=13,fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")

def plot_confidence(probs, preds, labels, path):
    fig,axes = plt.subplots(1,len(CLASSES),figsize=(16,4),sharey=True)
    fig.suptitle('Confidence Distribution per Class',fontsize=13,fontweight='bold')
    for i,(cls,ax) in enumerate(zip(CLASSES,axes)):
        mask = labels==i
        if mask.sum()==0: ax.set_title(cls); continue
        conf = probs[mask,i]
        correct = preds[mask]==i
        ax.hist(conf[correct],  bins=10,alpha=0.7,color='#2ECC71',label='Correct')
        ax.hist(conf[~correct], bins=10,alpha=0.7,color='#E74C3C',label='Wrong')
        ax.set_title(cls,fontsize=10); ax.set_xlabel('Confidence')
        if i==0: ax.set_ylabel('Count')
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")

def plot_f1_bar(report, acc, path):
    f1s    = [report.get(c,{}).get('f1-score',0) for c in CLASSES]
    colors = ['#2ECC71' if f>=0.8 else '#F39C12' if f>=0.6 else '#E74C3C' for f in f1s]
    fig,ax = plt.subplots(figsize=(9,5))
    bars = ax.bar(CLASSES, f1s, color=colors, edgecolor='white')
    ax.axhline(acc,color='#2C3E50',linestyle='--',linewidth=2,label=f'Overall Acc={acc:.2%}')
    ax.set_ylim(0,1.1); ax.set_ylabel('F1-Score',fontsize=12)
    ax.set_title('Per-Class F1  |  Green≥0.8  Orange≥0.6  Red<0.6',fontsize=11,fontweight='bold')
    for bar,v in zip(bars,f1s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{v:.2f}',ha='center',va='bottom',fontsize=11,fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches='tight'); plt.close()
    print(f"  Saved: {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ds     = TestDataset(args.jpg_root, args.annotation,
                         args.slow_frames, args.fast_frames)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.n_workers, collate_fn=collate_fn)
    model  = load_model(args.checkpoint)

    print("\n⏳ Running inference...")
    preds, labels, probs, names = run_inference(model, loader)

    acc = (preds==labels).mean()
    print(f"\n{'='*50}")
    print(f"  Overall Accuracy : {acc*100:.2f}%")
    print(f"  Total videos     : {len(preds)}")
    print(f"  Correct          : {(preds==labels).sum()}")
    print(f"{'='*50}\n")

    if HAS_SK:
        report     = classification_report(labels,preds,target_names=CLASSES,output_dict=True)
        report_str = classification_report(labels,preds,target_names=CLASSES)
        cm         = confusion_matrix(labels,preds)
        print(report_str)

        with open(os.path.join(args.result_path,'classification_report.txt'),'w') as f:
            f.write(f"Overall Accuracy: {acc*100:.2f}%\n\n{report_str}")

        print("\n📊 Saving graphs...")
        plot_confusion_matrix(cm, os.path.join(args.result_path,'1_confusion_matrix.png'))
        plot_per_class_metrics(report, os.path.join(args.result_path,'2_per_class_metrics.png'))
        plot_confidence(probs,preds,labels, os.path.join(args.result_path,'3_confidence.png'))
        plot_f1_bar(report, acc, os.path.join(args.result_path,'4_f1_summary.png'))

    # CSV
    csv_path = os.path.join(args.result_path,'per_video_results.csv')
    with open(csv_path,'w') as f:
        f.write('video,true_label,predicted_label,correct,' +
                ','.join([f'prob_{c}' for c in CLASSES]) + '\n')
        for n,l,p,pr in zip(names,labels,preds,probs):
            f.write(f"{n},{CLASSES[l]},{CLASSES[p]},{int(l==p)},"
                    + ','.join([f'{x:.4f}' for x in pr]) + '\n')
    print(f"  Saved: {csv_path}")
    print(f"\n✅ All results in: {args.result_path}")
