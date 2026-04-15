"""
================================================================
 DEEP LEARNING PRACTICALS — ALZHEIMER'S MRI DATASET
 All 8 Experiments Combined in One File
 Dataset: 4 classes — MildDemented, ModerateDemented,
          NonDemented, VeryMildDemented (2560 images each)
================================================================
 CHANGE THIS PATH TO YOUR DATASET FOLDER:
"""
DATA_DIR = "dataset"   # <-- update this
"""
================================================================
"""

# ── Shared Imports ───────────────────────────────────────────
import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, mean_squared_error)
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Dense, Input, Dropout, Conv2D,
                                      MaxPooling2D, GlobalAveragePooling2D,
                                      Flatten, LSTM, BatchNormalization)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         Callback)
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings("ignore")

CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# ── Shared Dataset Loader ────────────────────────────────────
def load_dataset(data_dir, image_size=(64, 64), grayscale=True,
                 max_per_class=None):
    X, y = [], []
    mode = "L" if grayscale else "RGB"
    for label in CLASSES:
        class_path = os.path.join(data_dir, label)
        if not os.path.exists(class_path):
            print(f"[WARNING] Not found: {class_path}")
            continue
        files = os.listdir(class_path)
        if max_per_class:
            files = files[:max_per_class]
        for fname in files:
            try:
                img = Image.open(os.path.join(class_path, fname))\
                           .convert(mode).resize(image_size)
                arr = np.array(img, dtype=np.float32) / 255.0
                X.append(arr.flatten())
                y.append(label)
            except Exception:
                pass
    return np.array(X, dtype=np.float32), np.array(y)

print("=" * 60)
print("  DL PRACTICALS — ALZHEIMER'S MRI DATASET")
print("=" * 60)

# ════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Perceptron (Binary classifier from scratch)
# ════════════════════════════════════════════════════════════════
print("\n\n>>> EXPERIMENT 1: Perceptron Learning Algorithm <<<")

print("Loading data (64x64 grayscale, 2 classes)...")
X_all, y_all = load_dataset(DATA_DIR, image_size=(64, 64))
le_main = LabelEncoder()
le_main.fit(y_all)

# Binary subset: NonDemented vs MildDemented
mask = np.isin(y_all, ["NonDemented", "MildDemented"])
X_bin = X_all[mask]
y_bin = (y_all[mask] == "MildDemented").astype(int)
X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
    X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

class Perceptron:
    def __init__(self, lr=0.01, epochs=50):
        self.lr, self.epochs = lr, epochs
        self.weights = self.bias = None
        self.errors_ = []

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for ep in range(self.epochs):
            errs = 0
            for xi, yi in zip(X, y):
                pred = 1 if np.dot(xi, self.weights) + self.bias >= 0 else 0
                upd = self.lr * (yi - pred)
                self.weights += upd * xi
                self.bias += upd
                errs += int(upd != 0)
            self.errors_.append(errs)
            if (ep + 1) % 10 == 0:
                print(f"  Epoch {ep+1}/{self.epochs} | Errors: {errs}")

    def predict(self, X):
        return (X @ self.weights + self.bias >= 0).astype(int)

p = Perceptron(lr=0.01, epochs=50)
p.fit(X_tr1, y_tr1)
y_pred1 = p.predict(X_te1)
print(f"\nExp1 Test Accuracy: {accuracy_score(y_te1, y_pred1):.4f}")
print(classification_report(y_te1, y_pred1,
      target_names=["NonDemented", "MildDemented"]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(p.errors_, color='crimson')
axes[0].set_title("Exp1 — Perceptron Errors/Epoch")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Errors"); axes[0].grid(True)
cm1 = confusion_matrix(y_te1, y_pred1)
sns.heatmap(cm1, annot=True, fmt='d', ax=axes[1], cmap="Blues",
            xticklabels=["NonDem", "Mild"], yticklabels=["NonDem", "Mild"])
axes[1].set_title("Exp1 — Confusion Matrix")
plt.tight_layout(); plt.savefig("exp1_perceptron.png", dpi=120); plt.show()


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — MLP (Non-linear data, decision boundary via PCA)
# ════════════════════════════════════════════════════════════════
print("\n\n>>> EXPERIMENT 2: MLP — Non-linear Classification <<<")

scaler2 = StandardScaler()
X_sc2 = scaler2.fit_transform(X_all)
y_enc2 = le_main.transform(y_all)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_sc2, y_enc2, test_size=0.2, random_state=42, stratify=y_enc2)

mlp2 = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu',
                      solver='adam', max_iter=100, random_state=42,
                      early_stopping=True, validation_fraction=0.1,
                      verbose=False)
mlp2.fit(X_tr2, y_tr2)
y_pred2 = mlp2.predict(X_te2)
print(f"Exp2 Test Accuracy: {accuracy_score(y_te2, y_pred2):.4f}")
print(classification_report(y_te2, y_pred2, target_names=le_main.classes_))

# Decision boundary (PCA 2D)
pca2 = PCA(n_components=2, random_state=42)
X_2d = pca2.fit_transform(X_sc2)
X_tr2d, X_te2d, y_tr2d, y_te2d = train_test_split(
    X_2d, y_enc2, test_size=0.2, random_state=42, stratify=y_enc2)
mlp2d = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100,
                       random_state=42).fit(X_tr2d, y_tr2d)
h = 0.5
xx, yy = np.meshgrid(
    np.arange(X_2d[:,0].min()-1, X_2d[:,0].max()+1, h),
    np.arange(X_2d[:,1].min()-1, X_2d[:,1].max()+1, h))
Z = mlp2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(mlp2.loss_curve_, label='Train', color='blue')
axes[0].set_title("Exp2 — MLP Loss Curve"); axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(True)
axes[1].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
colors2 = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4']
for i, cls in enumerate(le_main.classes_):
    m = y_te2d == i
    axes[1].scatter(X_te2d[m,0], X_te2d[m,1], label=cls[:8], s=15, alpha=0.7)
axes[1].set_title("Exp2 — Decision Boundary (PCA 2D)")
axes[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("exp2_mlp.png", dpi=120); plt.show()


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — ANN (EDA + multiple architectures + optimizers)
# ════════════════════════════════════════════════════════════════
print("\n\n>>> EXPERIMENT 3: ANN — EDA + Architecture/Optimizer Comparison <<<")

# --- EDA ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = pd.Series(y_all).value_counts()
axes[0].bar(counts.index, counts.values,
            color=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4'])
axes[0].set_title("Exp3 — Class Distribution")
axes[0].tick_params(axis='x', rotation=20)
for i, cls in enumerate(CLASSES):
    axes[1].hist(X_all[y_all==cls].mean(axis=1), bins=40, alpha=0.5,
                 label=cls[:8])
axes[1].set_title("Exp3 — Mean Pixel Intensity by Class")
axes[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("exp3_eda.png", dpi=120); plt.show()

# Preprocess for Keras
y_cat3 = to_categorical(le_main.transform(y_all), 4)
X_sc3 = StandardScaler().fit_transform(X_all)
X_tr3, X_te3, y_tr3, y_te3 = train_test_split(
    X_sc3, y_cat3, test_size=0.2, random_state=42,
    stratify=le_main.transform(y_all))
X_tr3, X_va3, y_tr3, y_va3 = train_test_split(
    X_tr3, y_tr3, test_size=0.1, random_state=42)
IN3 = X_tr3.shape[1]

cb3 = [EarlyStopping(monitor='val_loss', patience=8,
                      restore_best_weights=True)]

def build_ann(arch, opt='adam'):
    m = Sequential([Input(shape=(IN3,))] +
                   [Dense(u, activation='relu') for u in arch] +
                   [Dense(4, activation='softmax')])
    m.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m

archs = {"Shallow(64)": [64], "Medium(128-64)": [128,64],
         "Deep(256-128-64)": [256,128,64]}
arch_accs = {}
for name, arch in archs.items():
    m = build_ann(arch)
    m.fit(X_tr3, y_tr3, epochs=40, batch_size=64,
          validation_data=(X_va3, y_va3), callbacks=cb3, verbose=0)
    _, acc = m.evaluate(X_te3, y_te3, verbose=0)
    arch_accs[name] = acc
    print(f"  {name}: {acc:.4f}")

opts = {"Adam": Adam(0.001), "SGD": SGD(0.01), "RMSprop": RMSprop(0.001)}
opt_accs = {}
for name, opt in opts.items():
    m = build_ann([256,128,64], opt=opt)
    m.fit(X_tr3, y_tr3, epochs=40, batch_size=64,
          validation_data=(X_va3, y_va3), callbacks=cb3, verbose=0)
    _, acc = m.evaluate(X_te3, y_te3, verbose=0)
    opt_accs[name] = acc
    print(f"  Optimizer {name}: {acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(arch_accs.keys(), arch_accs.values(),
            color=['#FF6B6B','#4ECDC4','#45B7D1'])
axes[0].set_title("Exp3 — Architecture Comparison"); axes[0].set_ylim(0,1)
axes[0].tick_params(axis='x', rotation=15)
axes[1].bar(opt_accs.keys(), opt_accs.values(),
            color=['#FF9F43','#54A0FF','#5F27CD'])
axes[1].set_title("Exp3 — Optimizer Comparison"); axes[1].set_ylim(0,1)
plt.tight_layout(); plt.savefig("exp3_ann.png", dpi=120); plt.show()

# Save best model
best_ann = build_ann([256,128,64])
best_ann.fit(X_tr3, y_tr3, epochs=80, batch_size=64,
             validation_data=(X_va3, y_va3), callbacks=cb3, verbose=0)
best_ann.save("alzheimer_ann.h5")
print("Exp3 model saved: alzheimer_ann.h5")


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — NN from Scratch (6 optimizers, 3 batch strategies)
# ════════════════════════════════════════════════════════════════
print("\n\n>>> EXPERIMENT 4: NN from Scratch — Optimizers & Batching <<<")

# Load smaller subset for speed (from-scratch = slow)
X4, y4_lbl = load_dataset(DATA_DIR, image_size=(28,28), max_per_class=400)
y4_enc = le_main.transform(y4_lbl)
def one_hot(y, n): r=np.zeros((len(y),n)); r[np.arange(len(y)),y]=1; return r
y4_oh = one_hot(y4_enc, 4)
X_tr4, X_te4, y_tr4, y_te4 = train_test_split(
    X4, y4_oh, test_size=0.2, random_state=42, stratify=y4_enc)

def relu(z): return np.maximum(0, z)
def relu_d(z): return (z>0).astype(float)
def softmax(z):
    e=np.exp(z-np.max(z,axis=1,keepdims=True)); return e/e.sum(axis=1,keepdims=True)

class NNScratch:
    def __init__(self, sizes, opt='adam', lr=0.01):
        self.opt, self.lr = opt, lr
        self.W=[np.random.randn(sizes[i],sizes[i+1])*np.sqrt(2/sizes[i])
                for i in range(len(sizes)-1)]
        self.b=[np.zeros((1,sizes[i+1])) for i in range(len(sizes)-1)]
        self.vW=[np.zeros_like(w) for w in self.W]
        self.vb=[np.zeros_like(b) for b in self.b]
        self.sW=[np.zeros_like(w) for w in self.W]
        self.sb=[np.zeros_like(b) for b in self.b]
        self.GW=[np.zeros_like(w) for w in self.W]
        self.Gb=[np.zeros_like(b) for b in self.b]
        self.t=0; self.eps=1e-8; self.beta1=0.9; self.beta2=0.999

    def forward(self, X):
        self.A=[X]; self.Z=[]
        a=X
        for i,(w,b) in enumerate(zip(self.W,self.b)):
            z=a@w+b; self.Z.append(z)
            a=relu(z) if i<len(self.W)-1 else softmax(z)
            self.A.append(a)
        return self.A[-1]

    def backward(self, y):
        n=y.shape[0]; gW=[]; gb=[]; d=(self.A[-1]-y)/n
        for i in reversed(range(len(self.W))):
            gW.insert(0, self.A[i].T@d)
            gb.insert(0, d.sum(0,keepdims=True))
            if i>0: d=(d@self.W[i].T)*relu_d(self.Z[i-1])
        return gW,gb

    def update(self, gW, gb):
        self.t+=1
        for i in range(len(self.W)):
            gw,gb_=gW[i],gb[i]
            if self.opt=='gd':
                self.W[i]-=self.lr*gw; self.b[i]-=self.lr*gb_
            elif self.opt=='momentum':
                self.vW[i]=0.9*self.vW[i]+self.lr*gw
                self.vb[i]=0.9*self.vb[i]+self.lr*gb_
                self.W[i]-=self.vW[i]; self.b[i]-=self.vb[i]
            elif self.opt=='nesterov':
                vp=self.vW[i].copy(); vpb=self.vb[i].copy()
                self.vW[i]=0.9*self.vW[i]+self.lr*gw
                self.vb[i]=0.9*self.vb[i]+self.lr*gb_
                self.W[i]-=(-0.9*vp+(1.9)*self.vW[i])
                self.b[i]-=(-0.9*vpb+(1.9)*self.vb[i])
            elif self.opt=='adagrad':
                self.GW[i]+=gw**2; self.Gb[i]+=gb_**2
                self.W[i]-=self.lr*gw/(np.sqrt(self.GW[i])+self.eps)
                self.b[i]-=self.lr*gb_/(np.sqrt(self.Gb[i])+self.eps)
            elif self.opt=='rmsprop':
                self.sW[i]=0.9*self.sW[i]+0.1*gw**2
                self.sb[i]=0.9*self.sb[i]+0.1*gb_**2
                self.W[i]-=self.lr*gw/(np.sqrt(self.sW[i])+self.eps)
                self.b[i]-=self.lr*gb_/(np.sqrt(self.sb[i])+self.eps)
            elif self.opt=='adam':
                self.vW[i]=self.beta1*self.vW[i]+(1-self.beta1)*gw
                self.vb[i]=self.beta1*self.vb[i]+(1-self.beta1)*gb_
                self.sW[i]=self.beta2*self.sW[i]+(1-self.beta2)*gw**2
                self.sb[i]=self.beta2*self.sb[i]+(1-self.beta2)*gb_**2
                vc=self.vW[i]/(1-self.beta1**self.t)
                vbc=self.vb[i]/(1-self.beta1**self.t)
                sc=self.sW[i]/(1-self.beta2**self.t)
                sbc=self.sb[i]/(1-self.beta2**self.t)
                self.W[i]-=self.lr*vc/(np.sqrt(sc)+self.eps)
                self.b[i]-=self.lr*vbc/(np.sqrt(sbc)+self.eps)

    def train(self, X, y, epochs=25, strategy='mini-batch', bs=64):
        losses=[]
        for ep in range(epochs):
            n=X.shape[0]; loss=0
            if strategy=='batch':
                yp=self.forward(X)
                loss=-np.mean(y*np.log(yp+1e-9))
                gW,gb=self.backward(y); self.update(gW,gb)
            elif strategy=='sgd':
                idx=np.random.permutation(n)[:150]
                for i in idx:
                    yp=self.forward(X[i:i+1])
                    loss+=-np.mean(y[i:i+1]*np.log(yp+1e-9))
                    gW,gb=self.backward(y[i:i+1]); self.update(gW,gb)
                loss/=150
            else:  # mini-batch
                idx=np.random.permutation(n); nb=0
                for s in range(0,n,bs):
                    xi=X[idx[s:s+bs]]; yi=y[idx[s:s+bs]]
                    yp=self.forward(xi)
                    loss+=-np.mean(yi*np.log(yp+1e-9))
                    gW,gb=self.backward(yi); self.update(gW,gb); nb+=1
                loss/=nb
            losses.append(loss)
            if (ep+1)%5==0:
                acc=accuracy_score(np.argmax(y,1),
                    np.argmax(self.forward(X),1))
                print(f"  Ep{ep+1} loss={loss:.4f} acc={acc:.4f}")
        return losses

SIZES=[784,128,64,4]
OPTS=['gd','momentum','nesterov','adagrad','rmsprop','adam']
OPT_LABELS=['GD','Momentum','Nesterov','Adagrad','RMSProp','Adam']

print("\n-- 6 Optimizers (mini-batch) --")
opt4_losses={}; opt4_accs={}
for opt,lbl in zip(OPTS,OPT_LABELS):
    print(f" Optimizer: {lbl}")
    nn=NNScratch(SIZES, opt=opt, lr=0.01)
    losses=nn.train(X_tr4, y_tr4, epochs=25, strategy='mini-batch')
    acc=accuracy_score(np.argmax(y_te4,1), np.argmax(nn.forward(X_te4),1))
    opt4_losses[lbl]=losses; opt4_accs[lbl]=acc
    print(f"  Test Acc: {acc:.4f}")

print("\n-- 3 Batch Strategies (Adam) --")
STRATS=['batch','sgd','mini-batch']
SLABELS=['Batch GD','SGD','Mini-batch']
strat_losses={}; strat_accs={}
for st,sl in zip(STRATS,SLABELS):
    print(f" Strategy: {sl}")
    nn=NNScratch(SIZES, opt='adam', lr=0.01)
    losses=nn.train(X_tr4, y_tr4, epochs=25, strategy=st)
    acc=accuracy_score(np.argmax(y_te4,1), np.argmax(nn.forward(X_te4),1))
    strat_losses[sl]=losses; strat_accs[sl]=acc
    print(f"  Test Acc: {acc:.4f}")

colors4=['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FF9F43','#5F27CD']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i,(lbl,ls) in enumerate(opt4_losses.items()):
    axes[0,0].plot(ls, label=lbl, color=colors4[i])
axes[0,0].set_title("Exp4 — Loss (6 Optimizers)"); axes[0,0].legend(fontsize=7)
axes[0,0].set_xlabel("Epoch"); axes[0,0].grid(True)
axes[0,1].bar(opt4_accs.keys(), opt4_accs.values(), color=colors4)
axes[0,1].set_title("Exp4 — Optimizer Test Accuracy")
axes[0,1].set_ylim(0,1); axes[0,1].tick_params(axis='x', rotation=15)
for i,(sl,ls) in enumerate(strat_losses.items()):
    axes[1,0].plot(ls, label=sl, color=['#E74C3C','#3498DB','#2ECC71'][i], lw=2)
axes[1,0].set_title("Exp4 — Loss (3 Batch Strategies)")
axes[1,0].legend(); axes[1,0].grid(True)
axes[1,1].bar(strat_accs.keys(), strat_accs.values(),
              color=['#E74C3C','#3498DB','#2ECC71'])
axes[1,1].set_title("Exp4 — Batch Strategy Test Accuracy"); axes[1,1].set_ylim(0,1)
plt.tight_layout(); plt.savefig("exp4_nn_scratch.png", dpi=120); plt.show()


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 5 — ANN Regularization (L1, L2, Dropout, Early Stop)
# ════════════════════════════════════════════════════════════════
print("\n\n>>> EXPERIMENT 5: Regularization — L1, L2, Dropout, Early Stopping <<<")

IN5=IN3
def build_reg(reg=None, dr=0.0):
    kr=(l1(0.001) if reg=='l1' else l2(0.001) if reg=='l2'
        else l1_l2(0.0005,0.0005) if reg=='l1l2' else None)
    layers=[Input(shape=(IN5,)),
            Dense(256,activation='relu',kernel_regularizer=kr)]
    if dr>0: layers.append(Dropout(dr))
    layers+=[Dense(128,activation='relu',kernel_regularizer=kr)]
    if dr>0: layers.append(Dropout(dr))
    layers+=[Dense(64,activation='relu',kernel_regularizer=kr),
             Dense(4,activation='softmax')]
    m=Sequential(layers)
    m.compile(Adam(0.001),'categorical_crossentropy',metrics=['accuracy'])
    return m

configs5={
    "Baseline":       dict(reg=None, dr=0.0, es=False),
    "L1":             dict(reg='l1',  dr=0.0, es=False),
    "L2":             dict(reg='l2',  dr=0.0, es=False),
    "L1+L2":          dict(reg='l1l2',dr=0.0, es=False),
    "Dropout(0.3)":   dict(reg=None, dr=0.3, es=False),
    "Dropout(0.5)":   dict(reg=None, dr=0.5, es=False),
    "Early Stop":     dict(reg=None, dr=0.0, es=True),
    "L2+Drop+ES":     dict(reg='l2',  dr=0.3, es=True),
}
res5={}
for name,cfg in configs5.items():
    print(f"  Training: {name}")
    m=build_reg(cfg['reg'],cfg['dr'])
    cbs=[EarlyStopping('val_loss',patience=10,restore_best_weights=True)]\
        if cfg['es'] else []
    t0=time.time()
    hist=m.fit(X_tr3,y_tr3,epochs=50,batch_size=64,
               validation_data=(X_va3,y_va3),callbacks=cbs,verbose=0)
    _,acc=m.evaluate(X_te3,y_te3,verbose=0)
    res5[name]={'acc':acc,'hist':hist,'time':time.time()-t0,
                'ep':len(hist.history['loss'])}
    print(f"    Acc={acc:.4f} | Epochs={res5[name]['ep']}")

fig,axes=plt.subplots(2,2,figsize=(14,10))
clrs5=['#E74C3C','#3498DB','#2ECC71','#F39C12','#9B59B6','#1ABC9C','#E67E22','#34495E']
for i,(n,r) in enumerate(res5.items()):
    axes[0,0].plot(r['hist'].history['val_loss'],label=n,color=clrs5[i],lw=1.5)
axes[0,0].set_title("Exp5 — Val Loss Curves"); axes[0,0].legend(fontsize=6)
axes[0,0].set_xlabel("Epoch"); axes[0,0].grid(True)
names5=list(res5.keys()); accs5=[res5[n]['acc'] for n in names5]
axes[0,1].barh(names5,accs5,color=clrs5)
axes[0,1].set_title("Exp5 — Test Accuracy by Regularization")
axes[0,1].set_xlim(0,1)
axes[1,0].bar(names5,[res5[n]['time'] for n in names5],color=clrs5)
axes[1,0].set_title("Exp5 — Training Time (s)")
axes[1,0].tick_params(axis='x',rotation=30)
axes[1,1].bar(names5,[res5[n]['ep'] for n in names5],color=clrs5)
axes[1,1].set_title("Exp5 — Epochs Run")
axes[1,1].tick_params(axis='x',rotation=30)
plt.tight_layout(); plt.savefig("exp5_regularization.png", dpi=120); plt.show()

best5=max(res5,key=lambda k:res5[k]['acc'])
m5f=build_reg(configs5[best5]['reg'],configs5[best5]['dr'])
m5f.fit(X_tr3,y_tr3,epochs=80,batch_size=64,
        validation_data=(X_va3,y_va3),
        callbacks=[EarlyStopping('val_loss',patience=10,restore_best_weights=True)],
        verbose=0)
m5f.save("alzheimer_regularized.h5")
print(f"Exp5 best config: {best5} | model saved: alzheimer_regularized.h5")


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 6A — CNN Forward Propagation from Scratch
# ════════════════════════════════════════════════════════════════
print("\n\n>>> EXPERIMENT 6A: CNN Forward Propagation from Scratch <<<")
print("Settings: Stride=1, Padding=No, LR=0.01, Loss=MSE")

def conv2d(img, k, stride=1):
    kH,kW=k.shape; iH,iW=img.shape
    oH=(iH-kH)//stride+1; oW=(iW-kW)//stride+1
    out=np.zeros((oH,oW))
    for i in range(oH):
        for j in range(oW):
            out[i,j]=np.sum(img[i*stride:i*stride+kH,
                                j*stride:j*stride+kW]*k)
    return out

def maxpool(fm, ps=2, st=2):
    oH=(fm.shape[0]-ps)//st+1; oW=(fm.shape[1]-ps)//st+1
    out=np.zeros((oH,oW))
    for i in range(oH):
        for j in range(oW):
            out[i,j]=np.max(fm[i*st:i*st+ps, j*st:j*st+ps])
    return out

def mse(pred, tgt): return np.mean((pred-tgt)**2)
def sm(z): e=np.exp(z-np.max(z)); return e/e.sum()

# Load 1 sample per class
samples6a={}
for cls in CLASSES:
    p=os.path.join(DATA_DIR,cls)
    if os.path.exists(p):
        for f in os.listdir(p)[:1]:
            try:
                img=np.array(Image.open(os.path.join(p,f))
                             .convert("L").resize((64,64)),
                             dtype=np.float32)/255.0
                samples6a[cls]=img
            except: pass

filters6a=[np.random.randn(3,3)*0.1 for _ in range(8)]
W6a=None; b6a=None

print("\nForward pass through each sample:")
for cls,img in samples6a.items():
    fm1=[np.maximum(0,conv2d(img,f,stride=1)) for f in filters6a]
    pool1=[maxpool(fm) for fm in fm1]
    flat=np.concatenate([p.flatten() for p in pool1])
    if W6a is None:
        W6a=np.random.randn(flat.shape[0],128)*0.01; b6a=np.zeros(128)
        W6b=np.random.randn(128,4)*0.01; b6b=np.zeros(4)
    h=np.maximum(0, flat@W6a+b6a)
    out=sm(h@W6b+b6b)
    tgt=np.eye(4)[CLASSES.index(cls)]
    loss=mse(out,tgt)
    print(f"  {cls}: pred={CLASSES[np.argmax(out)][:8]} | MSE={loss:.5f}")

# Feature map visualization
sample_img=list(samples6a.values())[0]
fig,axes=plt.subplots(2,8,figsize=(20,6))
axes[0,0].imshow(sample_img,cmap='gray')
axes[0,0].set_title("Input"); axes[0,0].axis('off')
for j in range(1,8): axes[0,j].axis('off')
for i,f in enumerate(filters6a):
    fm=np.maximum(0,conv2d(sample_img,f))
    axes[1,i].imshow(maxpool(fm),cmap='viridis')
    axes[1,i].set_title(f"Filter{i+1}"); axes[1,i].axis('off')
fig.suptitle("Exp6A — CNN Feature Maps (no padding, stride=1)",fontweight='bold')
plt.tight_layout(); plt.savefig("exp6a_cnn_forward.png", dpi=120); plt.show()


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 6B — CNN + Transfer Learning (VGG16, ResNet50, etc.)
# ════════════════════════════════════════════════════════════════
print("\n\n>>> EXPERIMENT 6B: CNN + Transfer Learning <<<")
print("Models: Custom CNN, VGG16, ResNet50, InceptionV3, MobileNetV2")

from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNetV2

def load_rgb(data_dir, size=(224,224)):
    X,y=[],[]
    for label in CLASSES:
        cp=os.path.join(data_dir,label)
        if not os.path.exists(cp): continue
        for fname in os.listdir(cp):
            try:
                img=Image.open(os.path.join(cp,fname)).convert("RGB").resize(size)
                X.append(np.array(img,dtype=np.float32)/255.0); y.append(label)
            except: pass
    return np.array(X,dtype=np.float32), np.array(y)

print("Loading RGB 224x224 dataset...")
X6b,y6b_lbl=load_rgb(DATA_DIR,(224,224))
y6b_enc=le_main.transform(y6b_lbl)
y6b_cat=to_categorical(y6b_enc,4)
X_tr6,X_te6,y_tr6,y_te6=train_test_split(
    X6b,y6b_cat,test_size=0.2,random_state=42,stratify=y6b_enc)
X_tr6,X_va6,y_tr6,y_va6=train_test_split(
    X_tr6,y_tr6,test_size=0.1,random_state=42)

def resize_batch(X, size):
    return np.array([np.array(Image.fromarray(
        (img*255).astype(np.uint8)).resize(size))/255.0
        for img in X], dtype=np.float32)

# Custom CNN (64x64)
X_tr6_64=resize_batch(X_tr6,(64,64))
X_va6_64=resize_batch(X_va6,(64,64))
X_te6_64=resize_batch(X_te6,(64,64))

custom_cnn6=Sequential([
    Input(shape=(64,64,3)),
    Conv2D(32,(3,3),activation='relu',padding='same'),
    BatchNormalization(), MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu',padding='same'),
    BatchNormalization(), MaxPooling2D(2,2),
    Conv2D(128,(3,3),activation='relu',padding='same'),
    BatchNormalization(), MaxPooling2D(2,2),
    GlobalAveragePooling2D(),
    Dense(128,activation='relu'), Dropout(0.5),
    Dense(4,activation='softmax')
])
custom_cnn6.compile(Adam(0.001),'categorical_crossentropy',metrics=['accuracy'])
cb6=[EarlyStopping('val_loss',patience=8,restore_best_weights=True)]
print("\nTraining Custom CNN...")
hist_cc=custom_cnn6.fit(X_tr6_64,y_tr6,epochs=30,batch_size=32,
                         validation_data=(X_va6_64,y_va6),
                         callbacks=cb6,verbose=1)
_,acc_cc=custom_cnn6.evaluate(X_te6_64,y_te6,verbose=0)
print(f"Custom CNN Accuracy: {acc_cc:.4f}")

def build_tl(base_fn, shape, fine_tune=15):
    base=base_fn(weights='imagenet',include_top=False,input_shape=shape)
    for layer in base.layers: layer.trainable=False
    for layer in base.layers[-fine_tune:]: layer.trainable=True
    x=GlobalAveragePooling2D()(base.output)
    x=Dense(128,activation='relu')(x); x=Dropout(0.5)(x)
    out=Dense(4,activation='softmax')(x)
    m=Model(base.input,out)
    m.compile(Adam(0.0001),'categorical_crossentropy',metrics=['accuracy'])
    return m

tl_models={
    "VGG16":      (VGG16,      (224,224,3), 15),
    "ResNet50":   (ResNet50,   (224,224,3), 20),
    "InceptionV3":(InceptionV3,(299,299,3), 20),
    "MobileNetV2":(MobileNetV2,(224,224,3), 20),
}
all6_accs={"Custom CNN":acc_cc}
all6_hists={"Custom CNN":hist_cc}
y6_preds={}

for mname,(base_fn,shape,ft) in tl_models.items():
    print(f"\nTraining {mname}...")
    sz=shape[:2]
    Xtr=resize_batch(X_tr6,sz); Xva=resize_batch(X_va6,sz); Xte=resize_batch(X_te6,sz)
    m=build_tl(base_fn,shape,ft)
    hist=m.fit(Xtr,y_tr6,epochs=20,batch_size=16,
               validation_data=(Xva,y_va6),callbacks=cb6,verbose=1)
    _,acc=m.evaluate(Xte,y_te6,verbose=0)
    all6_accs[mname]=acc; all6_hists[mname]=hist
    y6_preds[mname]=np.argmax(m.predict(Xte,verbose=0),axis=1)
    print(f"{mname} Accuracy: {acc:.4f}")

fig,axes=plt.subplots(1,2,figsize=(14,5))
mn6=list(all6_accs.keys()); ac6=list(all6_accs.values())
clrs6=['#2C3E50','#E74C3C','#3498DB','#2ECC71','#F39C12']
bars=axes[0].bar(mn6,ac6,color=clrs6)
axes[0].set_title("Exp6B — Model Comparison"); axes[0].set_ylim(0,1)
axes[0].tick_params(axis='x',rotation=15)
for bar,acc in zip(bars,ac6):
    axes[0].text(bar.get_x()+bar.get_width()/2,acc+0.01,
                 f'{acc:.3f}',ha='center',fontsize=9,fontweight='bold')
for i,(n,h) in enumerate(all6_hists.items()):
    axes[1].plot(h.history['val_accuracy'],label=n,color=clrs6[i],lw=2)
axes[1].set_title("Exp6B — Val Accuracy Curves"); axes[1].legend(fontsize=8)
axes[1].set_xlabel("Epoch"); axes[1].grid(True)
plt.tight_layout(); plt.savefig("exp6b_transfer.png", dpi=120); plt.show()

best6b=max(y6_preds,key=lambda k:all6_accs[k])
y6_true=np.argmax(y_te6,axis=1)
print(f"\nBest TL model: {best6b}")
print(classification_report(y6_true,y6_preds[best6b],target_names=le_main.classes_))


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 7 — Autoencoder (Image Regeneration)
# ════════════════════════════════════════════════════════════════
print("\n\n>>> EXPERIMENT 7: Autoencoder — Image Regeneration <<<")
print("Architecture: 784→128→64→32→64→128→784  (code_size=32)")

INPUT_SIZE7=784; CODE_SIZE7=32
X7,y7_lbl=load_dataset(DATA_DIR,image_size=(28,28))
y7_enc=le_main.transform(y7_lbl)
X_tr7,X_te7,y_tr7,y_te7=train_test_split(
    X7,y7_enc,test_size=0.2,random_state=42,stratify=y7_enc)

# Build Autoencoder exactly as per spec
inp7=Input(shape=(INPUT_SIZE7,))
h1=Dense(128,activation='relu')(inp7)
h2=Dense(64, activation='relu')(h1)
code7=Dense(CODE_SIZE7,activation='relu',name='bottleneck')(h2)
h3=Dense(64, activation='relu')(code7)
h4=Dense(128,activation='relu')(h3)
out7=Dense(INPUT_SIZE7,activation='sigmoid')(h4)

autoencoder=Model(inp7,out7)
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
encoder7=Model(inp7,code7)
autoencoder.summary()

cb7=[EarlyStopping('val_loss',patience=15,restore_best_weights=True)]
print("\nTraining Autoencoder...")
hist7=autoencoder.fit(X_tr7,X_tr7,epochs=100,batch_size=256,
                       validation_data=(X_te7,X_te7),
                       callbacks=cb7,verbose=1)

X_recon=autoencoder.predict(X_te7,verbose=0)
mse7=mean_squared_error(X_te7.flatten(),X_recon.flatten())
print(f"\nReconstruction MSE: {mse7:.6f}")

# Visualize originals vs reconstructed
n_show=8; idx7=np.random.choice(len(X_te7),n_show,replace=False)
fig,axes=plt.subplots(2,n_show,figsize=(20,5))
for i,idx in enumerate(idx7):
    axes[0,i].imshow(X_te7[idx].reshape(28,28),cmap='gray')
    axes[0,i].set_title(le_main.classes_[y_te7[idx]][:7],fontsize=6)
    axes[0,i].axis('off')
    axes[1,i].imshow(X_recon[idx].reshape(28,28),cmap='gray')
    axes[1,i].set_title("Recon",fontsize=6); axes[1,i].axis('off')
axes[0,0].set_ylabel("Original",fontsize=8)
axes[1,0].set_ylabel("Reconstructed",fontsize=8)
fig.suptitle("Exp7 — Autoencoder: Original vs Reconstructed",fontweight='bold')
plt.tight_layout(); plt.savefig("exp7_autoencoder.png", dpi=120); plt.show()

# Latent space PCA
encoded7=encoder7.predict(X_te7,verbose=0)
pca7=PCA(n_components=2,random_state=42)
lat7=pca7.fit_transform(encoded7)
fig,ax=plt.subplots(figsize=(7,5))
for i,cls in enumerate(le_main.classes_):
    m=y_te7==i
    ax.scatter(lat7[m,0],lat7[m,1],label=cls[:8],alpha=0.5,s=10)
ax.set_title("Exp7 — Latent Space (PCA)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("exp7_latent.png", dpi=120); plt.show()


# ════════════════════════════════════════════════════════════════
# EXPERIMENT 8 — LSTM (Sequence Prediction)
# ════════════════════════════════════════════════════════════════
print("\n\n>>> EXPERIMENT 8: LSTM — Sequence Prediction <<<")
print("Settings: timestamp=50, split=75%, 4 LSTM layers, 128 neurons")

TIMESTAMP8=50; SPLIT8=0.75; LSTM_N=128; LSTM_L=4

X8,y8_lbl=load_dataset(DATA_DIR,image_size=(28,28))
y8_enc=le_main.transform(y8_lbl).astype(np.float32)

def make_sequences(X,y,seq_len):
    Xs,ys=[],[]
    for i in range(len(X)-seq_len):
        Xs.append(X[i:i+seq_len]); ys.append(y[i+seq_len])
    return np.array(Xs),np.array(ys)

X8s,y8s=make_sequences(X8,y8_enc,TIMESTAMP8)
print(f"Sequence shape: {X8s.shape}")
sp=int(len(X8s)*SPLIT8)
X_tr8,X_te8=X8s[:sp],X8s[sp:]
y_tr8,y_te8=to_categorical(y8s[:sp].astype(int),4),\
             to_categorical(y8s[sp:].astype(int),4)

def build_lstm(ts,feat,units=128,layers=4):
    m=Sequential([Input(shape=(ts,feat))])
    for i in range(layers):
        m.add(LSTM(units,return_sequences=(i<layers-1),
                   name=f"LSTM_{i+1}"))
        m.add(Dropout(0.3))
    m.add(Dense(64,activation='relu'))
    m.add(Dense(4,activation='softmax'))
    m.compile(Adam(0.001),'categorical_crossentropy',metrics=['accuracy'])
    return m

model8=build_lstm(TIMESTAMP8,X8s.shape[2],LSTM_N,LSTM_L)
model8.summary()

cb8=[EarlyStopping('val_loss',patience=10,restore_best_weights=True)]
print("\nTraining LSTM...")
hist8=model8.fit(X_tr8,y_tr8,epochs=50,batch_size=32,
                  validation_split=0.1,callbacks=cb8,verbose=1)

_,acc8=model8.evaluate(X_te8,y_te8,verbose=0)
y_pred8=np.argmax(model8.predict(X_te8,verbose=0),axis=1)
y_true8=np.argmax(y_te8,axis=1)
print(f"\nExp8 Test Accuracy: {acc8:.4f}")
print(classification_report(y_true8,y_pred8,target_names=le_main.classes_))

# Experiment: different LSTM configs
cfgs8=[("1L-64",1,64),("2L-128",2,128),("4L-128 (spec)",4,128),("4L-256",4,256)]
exp8_accs={}
for name,nl,nu in cfgs8:
    print(f"  Config {name}...")
    m=build_lstm(TIMESTAMP8,X8s.shape[2],nu,nl)
    m.fit(X_tr8,y_tr8,epochs=20,batch_size=32,validation_split=0.1,
          callbacks=[EarlyStopping('val_loss',patience=6,restore_best_weights=True)],
          verbose=0)
    _,acc=m.evaluate(X_te8,y_te8,verbose=0)
    exp8_accs[name]=acc; print(f"    Acc={acc:.4f}")

fig,axes=plt.subplots(1,3,figsize=(18,5))
axes[0].plot(hist8.history['loss'],label='Train'); axes[0].plot(hist8.history['val_loss'],label='Val')
axes[0].set_title("Exp8 — LSTM Loss"); axes[0].legend(); axes[0].grid(True)
axes[1].plot(hist8.history['accuracy'],label='Train'); axes[1].plot(hist8.history['val_accuracy'],label='Val')
axes[1].set_title("Exp8 — LSTM Accuracy"); axes[1].legend(); axes[1].grid(True)
axes[2].bar(exp8_accs.keys(),exp8_accs.values(),
            color=['#E74C3C','#3498DB','#2ECC71','#F39C12'])
axes[2].set_title("Exp8 — LSTM Config Comparison"); axes[2].set_ylim(0,1)
axes[2].tick_params(axis='x',rotation=15)
plt.tight_layout(); plt.savefig("exp8_lstm.png", dpi=120); plt.show()

cm8=confusion_matrix(y_true8,y_pred8)
plt.figure(figsize=(7,5))
sns.heatmap(cm8,annot=True,fmt='d',xticklabels=le_main.classes_,
            yticklabels=le_main.classes_,cmap="Blues")
plt.title("Exp8 — LSTM Confusion Matrix"); plt.tight_layout()
plt.savefig("exp8_lstm_cm.png", dpi=120); plt.show()

# ════════════════════════════════════════════════════════════════
print("\n\n" + "="*60)
print("  ALL 8 EXPERIMENTS COMPLETE")
print("="*60)
print("Saved models: alzheimer_ann.h5, alzheimer_regularized.h5")
print("Saved plots : exp1_perceptron.png  through  exp8_lstm_cm.png")
