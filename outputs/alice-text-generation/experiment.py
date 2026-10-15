"""Reproducible word-level recurrent language-model experiment."""
from pathlib import Path
from collections import Counter
import argparse, csv, hashlib, json, math, platform, random, re, time, unicodedata
from datetime import datetime, timezone
from urllib.request import urlopen
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
SOURCE = 'https://www.gutenberg.org/ebooks/11'
DOWNLOAD = 'https://www.gutenberg.org/ebooks/11.txt.utf-8'
KINDS = ['SimpleRNN', 'BidirectionalRNN', 'LSTM', 'GRU']
CONFIG = dict(context=20, embedding=64, hidden=128, batch_size=64,
              learning_rate=0.001, epochs=25, patience=4, seed=42,
              temperatures=[0.5, 1.0, 1.5], sampling_seeds=[101, 202, 303],
              generated_words=100, min_frequency=2)

def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_text(raw):
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    body = re.split(r'\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n', raw, maxsplit=1)[-1]
    body = re.split(r'\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK', body, maxsplit=1)[0]
    # The opening narrative occurs after the title, credits and contents.
    start = re.search(r'Alice was beginning to get very tired', body, re.I)
    if not start: raise ValueError('Expected story opening missing; inspect the downloaded edition.')
    body = body[start.start():]
    body = re.sub(r'(?im)^[ \t]*CHAPTER[ \t]+[IVXLCDM]+\.[ \t]*\n(?:[ \t]*\n)*[^\n]+\n', '\n', body)
    body = re.sub(r'\[Illustration[^\]]*\]', '', body, flags=re.I)
    body = unicodedata.normalize('NFKC', body).translate(str.maketrans({'’': "'", '‘': "'", '—':' ', '–':' '}))
    words = re.findall(r"[a-z]+(?:'[a-z]+)*", body.lower())
    assert words[:5] == ['alice','was','beginning','to','get']
    assert 'gutenberg' not in words and 'chapter' not in words and len(words) > 20000
    return words

def prepare():
    raw_path = ROOT/'data/alice_raw.txt'
    raw_path.parent.mkdir(exist_ok=True)
    if not raw_path.exists():
        with urlopen(DOWNLOAD, timeout=60) as response: raw_path.write_bytes(response.read())
    raw = raw_path.read_bytes()
    provenance = ROOT/'data/provenance.json'
    if not provenance.exists():
        save_json(provenance, dict(source=SOURCE, download_url=DOWNLOAD,
            retrieved_utc=datetime.now(timezone.utc).isoformat(), sha256=hashlib.sha256(raw).hexdigest()))
    assert json.loads(provenance.read_text())['sha256'] == hashlib.sha256(raw).hexdigest()
    words = clean_text(raw.decode('utf-8-sig'))
    (ROOT/'data/alice_clean.txt').write_text(' '.join(words), encoding='utf-8')
    a,b = int(len(words)*.8),int(len(words)*.9)
    parts = dict(train=words[:a], validation=words[a:b], test=words[b:])
    counts = Counter(parts['train'])
    vocab = ['<unk>'] + sorted(w for w,c in counts.items() if c >= CONFIG['min_frequency'])
    lookup = {w:i for i,w in enumerate(vocab)}
    datasets, stats = {}, {}
    for split, tokens in parts.items():
        ids = torch.tensor([lookup.get(w,0) for w in tokens], dtype=torch.long)
        windows = ids.unfold(0, CONFIG['context']+1, 1)
        datasets[split] = TensorDataset(windows[:,:-1].contiguous(), windows[:,-1].contiguous())
        stats[split] = dict(words=len(tokens), sequences=len(windows), unknown_rate=(ids==0).float().mean().item())
    save_json(ROOT/'data/vocabulary.json', vocab)
    save_json(ROOT/'data/statistics.json', dict(total_words=len(words), vocabulary_size=len(vocab), splits=stats,
        split_boundaries=[0,a,b,len(words)], policy='chronological split before windowing; training-only vocabulary'))
    seeds = [parts['train'][offset:offset+20] for offset in [0,1000,2000]]
    save_json(ROOT/'data/seeds.json', seeds)
    return datasets, vocab, seeds

class TextModel(nn.Module):
    def __init__(self, kind, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, CONFIG['embedding'])
        self.bidirectional = kind == 'BidirectionalRNN'
        cls = {'SimpleRNN':nn.RNN,'BidirectionalRNN':nn.RNN,'LSTM':nn.LSTM,'GRU':nn.GRU}[kind]
        self.recurrent = cls(CONFIG['embedding'], CONFIG['hidden'], batch_first=True, bidirectional=self.bidirectional)
        self.classifier = nn.Linear(CONFIG['hidden']*(2 if self.bidirectional else 1), vocab_size)
    def forward(self, x):
        _, hidden = self.recurrent(self.embedding(x))
        if isinstance(hidden, tuple): hidden = hidden[0]
        context = torch.cat([hidden[-2],hidden[-1]],dim=1) if self.bidirectional else hidden[-1]
        return self.classifier(context)

def evaluate(model, loader, device):
    model.eval(); total_loss=correct=n=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); logits=model(x)
            total_loss += nn.functional.cross_entropy(logits,y,reduction='sum').item()
            correct += (logits.argmax(1)==y).sum().item(); n += len(y)
    return dict(loss=total_loss/n, perplexity=math.exp(total_loss/n), accuracy=correct/n)

def synchronize(device):
    if device.type=='cuda': torch.cuda.synchronize()

def generate(model, seed_words, vocab, temperature, rng_seed, device):
    lookup={w:i for i,w in enumerate(vocab)}
    context=[lookup.get(w,0) for w in seed_words]
    rng=torch.Generator(device='cpu').manual_seed(rng_seed)
    generated=[]; model.eval()
    with torch.no_grad():
        for _ in range(CONFIG['generated_words']):
            logits=model(torch.tensor([context[-20:]],device=device))[0].cpu()/temperature
            logits[0]=-float('inf')
            token=torch.multinomial(torch.softmax(logits,0),1,generator=rng).item()
            context.append(token); generated.append(vocab[token])
    assert len(generated)==100 and '<unk>' not in generated
    return generated

def diversity(words):
    bigrams=list(zip(words,words[1:])); trigrams=list(zip(words,words[1:],words[2:]))
    return dict(distinct_1=len(set(words))/len(words), distinct_2=len(set(bigrams))/len(bigrams),
                repeated_trigram_rate=1-len(set(trigrams))/len(trigrams))

def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def run(smoke=False):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    seed_all(CONFIG['seed'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=ROOT/'results'/('smoke' if smoke else 'full'); out.mkdir(parents=True,exist_ok=True)
    checkpoint_dir=ROOT/'checkpoints'/('smoke' if smoke else 'full'); checkpoint_dir.mkdir(parents=True,exist_ok=True)
    datasets,vocab,seeds=prepare()
    if smoke:
        datasets={k:TensorDataset(*(t[:128] for t in d.tensors)) for k,d in datasets.items()}
    save_json(out/'config.json', dict(CONFIG, mode='smoke' if smoke else 'full', actual_max_epochs=1 if smoke else CONFIG['epochs']))
    save_json(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        processor=platform.processor(), torch=torch.__version__, numpy=np.__version__, device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None, threads=torch.get_num_threads()))
    metrics=[]; histories={}; samples=[]
    for kind in KINDS:
        seed_all(CONFIG['seed'])
        model=TextModel(kind,len(vocab)).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate'])
        generator=torch.Generator().manual_seed(CONFIG['seed'])
        train=DataLoader(datasets['train'],batch_size=64,shuffle=True,generator=generator)
        val=DataLoader(datasets['validation'],batch_size=128)
        test=DataLoader(datasets['test'],batch_size=128)
        history=[]; best=float('inf'); stale=0; best_epoch=0
        synchronize(device); start=time.perf_counter()
        for epoch in range(1,(1 if smoke else CONFIG['epochs'])+1):
            model.train(); loss_sum=n=0
            for x,y in train:
                x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
                logits=model(x); loss=nn.functional.cross_entropy(logits,y)
                assert logits.shape==(len(y),len(vocab)) and torch.isfinite(loss)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
                loss_sum+=loss.item()*len(y); n+=len(y)
            validation=evaluate(model,val,device)
            history.append(dict(epoch=epoch,train_loss=loss_sum/n,validation_loss=validation['loss']))
            print(f'{kind} epoch {epoch}: train={loss_sum/n:.4f} val={validation["loss"]:.4f}',flush=True)
            if validation['loss'] < best:
                best=validation['loss']; best_epoch=epoch; stale=0
                torch.save(model.state_dict(),checkpoint_dir/f'{kind}.pt')
            else: stale+=1
            if stale>=CONFIG['patience']: break
        synchronize(device); elapsed=time.perf_counter()-start
        model.load_state_dict(torch.load(checkpoint_dir/f'{kind}.pt',map_location=device,weights_only=True))
        result=evaluate(model,test,device)
        metrics.append(dict(model=kind,parameters=sum(p.numel() for p in model.parameters()),
            training_seconds=elapsed,epochs=len(history),best_epoch=best_epoch,**result))
        histories[kind]=history
        for seed_index,seed_words in enumerate(seeds):
            for temperature in CONFIG['temperatures']:
                for rng_seed in CONFIG['sampling_seeds']:
                    words=generate(model,seed_words,vocab,temperature,rng_seed,device)
                    samples.append(dict(model=kind,seed_index=seed_index,temperature=temperature,
                        sampling_seed=rng_seed,prompt=' '.join(seed_words),generated=' '.join(words),**diversity(words)))
        write_csv(out/'metrics.csv',metrics); save_json(out/'histories.json',histories)
        save_json(out/'samples.json',samples)
    assert len(samples)==108
    write_csv(out/'samples.csv',samples)
    from reporting import build_report
    build_report(out)
    return metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser(); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true'); mode.add_argument('--full',action='store_true')
    args=parser.parse_args(); run(smoke=args.smoke)
