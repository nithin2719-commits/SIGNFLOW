@echo off
echo ============================================================
echo  ASL Sign Language Model Training
echo  3D Landmarks (x,y,z) - 8-block Transformer - Target 90%%+
echo ============================================================
echo.

cd /d "c:\Users\Asus\project\New Msasl"

echo [INFO] Listing training words and sample counts...
python -c "
from pathlib import Path
import os

data = Path(r'c:\Users\Asus\project\New Msasl\landmark_data_random_split')
if not data.exists():
    data = Path(r'c:\Users\Asus\project\New Msasl\landmark_data')

train_dir = data / 'train'
val_dir   = data / 'val'

PRIORITY = {
    'hi','hello','bye','good_morning','how_are_you','how','what','where','when',
    'who','why','yes','no','ok','fine','please','sorry','thankyou','my','name',
    'i','you','we','love','i_love_you','help','want','need','not_understand',
    'deaf','happy','sad','tired','hungry','mother','father','family','friend',
    'today','tomorrow','now','later','eat','drink','go','come','stop','finish',
    'understand','think','nice_to_meet_you','good','morning','night','welcome',
    'excited','angry','afraid','nervous','proud','lonely','confused','bored',
    'brother','sister','baby','child','children','husband','wife','uncle','aunt',
    'grandfather','grandmother','cousin','son','daughter','boyfriend','girlfriend',
    'man','woman','girl','boy','person','people','home','school','hospital',
    'water','food','dog','cat','phone','computer','book','money','music',
    'one','two','three','four','five','six','seven','eight','nine','ten',
    'red','blue','green','yellow','orange','black','white','brown','purple','pink',
    'big','small','hot','cold','old','new','fast','slow','beautiful','cute','right',
    'wrong','same','here','there','up','down','sick','work','learn','make','give',
    'take','talk','ask','tell','listen','read','write','play','wait','meet','buy',
    'see','feel','have','do','remember','forget','find','teach','use','try'
}

def is_num(n): return n.replace('_','').replace('-','').isdigit()

words = sorted([d.name for d in train_dir.iterdir() if d.is_dir() and not is_num(d.name)])
total_train = 0
total_val   = 0
kept = []
skipped = []

for w in words:
    n = len(list((train_dir / w).glob('*.npy')))
    is_p = w.lower() in PRIORITY
    thresh = 5 if is_p else 20
    if n >= thresh:
        kept.append((w, n, is_p))
        total_train += n
        vd = val_dir / w
        if vd.exists(): total_val += len(list(vd.glob('*.npy')))
    else:
        skipped.append(w)

print()
print('=' * 60)
print(f'WORDS TO TRAIN: {len(kept)}')
print(f'SKIPPED (too few samples): {len(skipped)}')
print(f'TOTAL TRAIN SAMPLES: {total_train}')
print(f'TOTAL VAL   SAMPLES: {total_val}')
print('=' * 60)
print()
print('--- ALL TRAINING WORDS ---')
for w, n, p in kept:
    tag = ' [PRIORITY]' if p else ''
    print(f'  {w}: {n} samples{tag}')
print()
print(f'TOTAL: {len(kept)} words')
print('=' * 60)
"

echo.
echo [INFO] Starting training now...
echo [INFO] This will take 6-10 hours. Do NOT close this window.
echo.

python train_common_words.py

echo.
echo ============================================================
echo  TRAINING COMPLETE!
echo  Copy model to overlay:
echo  python copy_models.py
echo ============================================================
pause
