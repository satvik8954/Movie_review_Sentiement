import os
import re
import math
import json
import random
import urllib.request
import tarfile
from collections import Counter,defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
#hyperparameters
vocab_size=8000
max_seq_len=256

embed_dim=256
num_heads=8
num_layers=4
ffn_dim=1024
dropout=0.1
num_classes=2

batch_size=32
lr=1e-4
epochs=5
device="cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device:{device}")

#tokenizer

class BPETokenizer:
    def __init__(self,vocab_size=8000):
        self.vocab_size=vocab_size
        self.special_tokens=["[PAD]","[UNK]","[CLS]","[SEP]"]
        self.PAD_ID=0
        self.UNK_ID=1
        self.CLS_ID=2
        self.SEP_ID=3

        self.merges={} #dic oftokena token b

        self.vocab={}# token->iint
        self.inv_vocab={} #int-> token
        self.trained=False

    def _get_word_freqs(self,texts):
        word_freq=Counter()
        for text in texts:
            words=re.findall(r'\b\w+\b',text.lower())
            for word in words:
                word_tuple=tuple(list(word)+[' '])
                word_freq[word_tuple]+=1
        return word_freq
    def _get_pair_freqs(self,word_freqs):
        pair_freqs=Counter()
        for wordd,freq in word_freqs.items():
            for i in range(len(wordd)-1):
                pair_freqs[wordd[i],wordd[i+1]]+=freq
        return pair_freqs
    def _merge_pair(self,word_freqs,best_pair):
        new_word_freqs={}
        bigram=best_pair
        for word,freq in word_freqs.items():
            new_word=[]
            i=0
            while i<len(word):
                if i<len(word)-1 and (word[i],word[i+1])==bigram:
                    new_word.append(word[i]+word[i+1])
                    i+=2
                else:
                    new_word.append(word[i])
                    i+=1
            new_word_freqs[tuple(new_word)]=freq
        return new_word_freqs
    
        
    def train(self,texts):
    
        print("Training BPE tokenizer...")
        words_freqs=self._get_word_freqs(texts)

        all_chars=set()
        for word in words_freqs:
            all_chars.update(word)
        vocab=list(self.special_tokens)+sorted(all_chars)
        num_merges=self.vocab_size-len(vocab)
        for i in range(num_merges):
            pair_freqs=self._get_pair_freqs(words_freqs)
            if not pair_freqs:
                break


            best_pair=max(pair_freqs,key=pair_freqs.get)
            merged=best_pair[0]+best_pair[1]
            self.merges[best_pair]=merged
            vocab.append(merged)
            words_freqs=self._merge_pair(words_freqs,best_pair)

            if((i+1)%500==0):
                print(f"Merge {i+1}/{num_merges} | vocab size: {len(vocab)}")

        self.vocab={tok:idx for idx , tok in enumerate(vocab)}
        self.inv_vocab={idx:tok for idx,tok in self.vocab.items()}
        self.trained=True
        print(f"BPE Traingin is done BHAAI  vocab size emo {len(self.vocab)}")
    def _tokenize_word(self,word):
        tokens=list(word)+["_"]
        for pair,merged in self.merges.items():
            i=0
            new_tokens=[]
            while i<len(tokens):
                if i<len(tokens)-1 and (tokens[i],tokens[i+1])==pair:
                    new_tokens.append(merged)
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            tokens=new_tokens
        return tokens
    def encode(self,text,max_length=None):
        words=re.findall(r'\b\w+\b',text.lower())
        tokens=[self.CLS_ID]
        for word in words:
            bpe_tokens=self.tokenize_word(word)
            for took in bpe_tokens:
                tokens.append(self.vocab.get(took,self.UNK_ID))

        if max_length:
            tokens=tokens[:max_length]
        return tokens
    def decode(self,ids):
        tokens=[self.inv_vocab.get(i,"[UNK]") for i in ids if i not in (self.PAD_ID,self.CLS_ID,self.SEP_ID)]
        return " ".join(t.replace("_ ","") for t in tokens)
    def save(self,path):
        data={
            "vocab":self.vocab,
            "merges":{f"{a}|||{b}": m for (a,b),m in self.merges.items()}

        }
        with open(path,"w") as f:
            json.dump(data,f)

    def load(self,path):
        with open(path) as f:
            data=json.load(f)
        self.vocab=data["vocab"]
        self.inv_vocab={int(v):k for k,v in self.vocab.items()}
        self.merges={
            tuple(k.split("|||")):v
            for k,v in data["merges"].items()

        }
        self.trained=True
def download_Data(data_dir="imdb_data"):
    url="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path=os.path.join(data_dir,"aclImdb_v1.tar.gz")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(os.path.join(data_dir,"aclImdb")):
        print("Donwload ithundhi bhAAi")
        urllib.request.urlretrieve(url,tar_path)
        print("Extract ithundi bhaai")
        with tarfile.open(tar_path) as tar:
            tar.extractall(data_dir)
        print("Done")
def load_imbd_split(data_dir,split="train",max_samples=None):
    texts,labels=[],[]
    base=os.path.join(data_dir,"aclImdb",split)
    for label_str,label_int in [("pos",1),("neg",0)]:
        folder=os.path.join(base,label_str)
        files=os.listdir(folder)
        if max_samples:
            files=files[:max_samples//2]
        for fname in files:
            with open(os.path.join(folder,fname),encoding="utf-8") as f:
                texts.append(f.read())
            labels.append(label_int)

        
    combined=list(zip(texts,labels))
    random.shuffle(combined)
    texts,labels=zip(*combined)
    return list(texts),list(labels)

class IMDBdataset(Dataset):
    def __init__(self,texts,labels,tokenizer,max_len):
        self.labels=labels
        self.max_len=max_len
        self.pad_id=tokenizer.PAD_ID

        print(f"Tokenizing {len(texts)} exmples")
        self.encoded=[tokenizer.encode(t,max_length=max_len)for t in texts]

    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        ids=self.encoded[idx]
        label=self.labels[idx]
        padding_len=self.max_len-len(ids)
        attention_mask=[1] *len(ids)+[0]*padding_len
        ids=ids+[self.pad_id]*padding_len
        return(torch.tensor(ids,dtype=torch.long),
               torch.tensor(attention_mask,dtype=torch.bool),
               torch.tensor(label,dtype=torch.long))
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout):
        super().__init__()
        assert embed_dim%num_heads==0 ,"embed_data must be divisble by num_heads"
        self.num_heads=num_heads
        self.head_dim=embed_dim//num_heads
        self.scale=self.head_dim**-0.5

        self.qkv=nn.Linear(embed_dim,3*embed_dim,bias=False)
        self.out_proj=nn.Linear(embed_dim,embed_dim)
        self.attn_drop=nn.Dropout(dropout)
        self.out_drop=nn.Dropout(dropout)

    def forward(self,x,padding_mask=None):
        B,T,C=x.shape

        qkv=self.qkv(x)
        q,k,v=qkv.split(C,dim=-1)

        def split_heads(t):
            return t.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        q,k,v=split_heads(q),split_heads(k),split_heads(v)

        attn=(q@k.transpose(-2,-1)*self.scale)

        if padding_mask is not None:
            mask=padding_mask.unsqueeze(1).unsqueeze(2)
            attn=attn.masked_fill(~mask,float("-inf"))
        attn=F.softmax(attn,dim=-1)
        attn=self.attn_drop(attn)
        out=attn@v
        out=out.transpose(1,2).contiguous().view(B,T,C)
        return self.out_drop(self.out_proj(out))
class FeedForward(nn.Module):
    def __init__(self,embed_dim,ffn_dim,dropout):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(embed_dim,ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim,embed_dim),
            nn.Dropout(dropout)
            )
    def forward(self,x):
        return self.net(x)
class TranformerEncoderBlock(nn.Module):
    def __init__(self,embed_dim,num_heads,ffn_dim,dropout):
        super().__init__()
        self.norm1=nn.LayerNorm(embed_dim)
        self.attn=MultiHeadSelfAttention(embed_dim,num_heads,dropout)
        self.norm2=nn.LayerNorm(embed_dim)
        self.ffn=FeedForward(embed_dim,ffn_dim,dropout)
        self.drop=nn.Dropout(dropout)

    def forward(self,x,padding_mask=None):
        x=x+self.attn(self.norm1(x),padding_mask)
        x=x+self.ffn(self.norm2(x))
        return x
    
class TranformerClassifer(nn.Module):
    def __init__(self,vocab_size,embed_dim,num_heads,num_layers,ffn_dim,max_seq_len,dropout,num_classes):
        super.__init__()
        self.tok_emb=nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.pos_emb=nn.Embedding(max_seq_len,embed_dim)
        self.drop=nn.Dropout(dropout)

        self.blocks=nn.ModuleList([TranformerEncoderBlock(embed_dim,num_heads,ffn_dim,dropout)
        for _ in range(num_layers)])
        self.norm=nn.LayerNorm(embed_dim)

        self.classifier=nn.Sequential(
            nn.Linear(embed_dim,embed_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim//2,num_classes)

        )
        self.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Embedding):
                nn.init.normal_(m.weight,std=0.02)
    def forward(self,input_ids,attention_mask):
        B,T=input_ids.shape
        tok_vectors=self.tok_emb(input_ids)
        positions=torch.arange(T,device=input_ids.device)
        pos_vectors=self.pos_emb(positions)
        x=self.drop(tok_vectors+pos_vectors)
        for block in self.blocks:
            x=block(x,attention_mask)
        x=self.norm(x)

        cls_output=x[:,0,:]
        logits=self.classifier(cls_output)
        return logits
    

def train_epoch(model,loader,optimizer,scheduler,device):
    model.train()
    total_loss,total_correct,total_samples=0,0,0
    for batch_idx,(ids,mask,labels) in enumerate(loader):
        ids,mask,lables=ids.to(device),mask.to(device),lables.to(device)

        logits=model(ids,mask)

        loss=F.cross_entropy(logits,labels)
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        scheduler.step()

        predictions=logits.argmax(dim=-1)
        total_correct+=(predictions==labels).sum().item()
        total_loss+=loss.item()*len(labels)
        total_samples+=len(labels)

        if(batch_idx+1)%50==0:
            running_acc=total_correct/total_samples
            running_loss=total_loss/total_samples
            print(F"batch {batch_idx+1}/{len(loader)} |"
                  f"loss: {running_loss:.4f}|acc: {running_acc:.4f}")
        return total_loss/total_samples,total_correct/total_samples
@torch.no_grad()
def evaluate(model,loader,device):
    model.eval()
    total_loss,total_correct,total_samples=0,0,0
    for idx,mask,labels in loader:
        ids,mask,labels =ids.to(device),mask.to(device),labels.to(device)
        logits=model(ids,mask)
        loss=F.cross_entropy(logits,labels)
        predictions=logits.argmax(dim=-1)
        total_correct+=(predictions==labels).sum().item()
        total_loss+=loss.item()*len(labels)
        total_samples+=len(labels)
    return total_loss/total_samples,total_correct/total_samples
@torch.no_grad()
def predict(model,tokenizer,text,device,max_len=max_seq_len):
    model.eval()
    ids=tokenizer.encode(text,max_length=max_len)
    mask=[1]*len(ids)+[0]*(max_len-len(ids))
    ids=ids+[tokenizer.PAD_ID]*(max_len-len(ids))
    
    ids_t=torch.tensor([ids] ,dtype=torch.long).to(device)
    mask_t=torch.tensor([mask] ,dtype=torch.bool).to(device)

    logits=model(ids_t,mask_t)
    probs=F.softmax(logits,dim=-1)[0]

    label='positive' if probs[1]>probs[0] else 'negative'
    conf=probs.max().item()
    return label,conf
    

if __name__=="__main__":
    random.seed(42)
    torch.manual_seed(42)

    download_Data()

    DATA_DIR="imdb_data"

    train_texts,train_labels=load_imbd_split(DATA_DIR,"train",max_samples=8000)
    test_texts,test_lables=load_imbd_split(DATA_DIR,"test",max_samples=2000)
    print(f"Train: {len(train_texts)} | test: {len(test_texts)}")

    TOKENIZER_PATH="bpe_Tokenizer.json"
    tokenizer=BPETokenizer(vocab_size=vocab_size)
    if os.path.exists(TOKENIZER_PATH):
        print("\nLoading save tokenizer...")
        tokenizer.load(TOKENIZER_PATH)
    else:
        print("\n Training BPE tokenizer on training data...")
        tokenizer.train(train_texts)
        tokenizer.save(TOKENIZER_PATH)

    actual_vocab_size=len(tokenizer.vocab)
    print(f"Vocabulary size: {actual_vocab_size}")

    train_dataset=IMDBdataset(train_texts,train_labels,tokenizer,max_seq_len)
    test_dataset=IMDBdataset(test_texts,test_lables,tokenizer,max_seq_len)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

    print("\nbuidling the MODEL BHAAAAI")
    model=TranformerClassifer(vocab_size=actual_vocab_size,embed_dim=embed_dim,num_heads=num_heads,num_layers=num_layers,ffn_dim=ffn_dim,max_seq_len=max_seq_len,dropout=dropout,num_classes=num_classes).to(device)

    num_params=sum(p.numel() for p in model.parameters( ) if p.requires_grad)
    print(f"Trainable parameters :{num_params:,}")

    optimizer=torch.optim.Adamw(model.parameters(),lr=lr,weight_decay=0.01,betas=(0.9,0.999))
    total_Steps=len(train_loader)*epochs
    warmup_steps=total_Steps//10
    def lr_lambda(step):
        if step<warmup_steps:
            return step/max(1,warmup_steps)
        progress=(step-warmup_steps)/max(1,total_Steps-warmup_steps)
        return 0.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)

    print(f"\n{'--'*60}")
    print(f'Training for {epochs} epochs on{device}')
    print(f"{'--'*60}")
    best_val_Acc=0.0
    for epoch in range(1,epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_losses,train_Acc=train_epoch(model,train_loader,optimizer,scheduler,device)
        val_losses,val_acc= evaluate(model,test_loader,device)
        print(f"\n train loss ochesi : {train_losses:.4f}| acc:{train_Acc:.4f}")
        print(f"val losses ochesi:{val_losses:.4f}| acc:{val_acc:.4f}")

        if val_acc>best_val_Acc:
            best_val_Acc=val_acc
            torch.save(model.state_dict(),"Best_classifer.pt")
            print(f" New best model ochesinduuuu {val_acc:.4f}")
        print(f"\n{'--'*60}")
        print("training over saar",best_val_Acc)
        print("sample prediction")
        examples=[
            "This movie was absolutely fantastic! The acting was superb and the story gripping.",
        "Terrible film. Boring, predictable, and a complete waste of time.",
        "It was okay, nothing special but not bad either.",
        "One of the greatest films I have ever seen. A true masterpiece.",
        ]
        for review in examples:
            label,confidence=predict(model,tokenizer,review,device)
            print(f"[{label.upper():8s}] {confidence:.0%} {review[:70]}...")




















        



    
    











             




    





