"""Checks for leakage boundaries, recurrent heads, checkpoint fidelity and sampling."""
import tempfile, unittest
from pathlib import Path
import torch
import experiment as e

class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.datasets,cls.vocab,cls.seeds=e.prepare()

    def test_training_only_vocabulary_and_boundaries(self):
        from collections import Counter
        raw=(e.ROOT/'data/alice_raw.txt').read_text(encoding='utf-8-sig')
        words=e.clean_text(raw); a,b=int(.8*len(words)),int(.9*len(words))
        expected=['<unk>']+sorted(w for w,c in Counter(words[:a]).items() if c>=2)
        self.assertEqual(self.vocab,expected)
        lookup={w:i for i,w in enumerate(self.vocab)}
        for key,part in zip(['train','validation','test'],[words[:a],words[a:b],words[b:]]):
            x,y=self.datasets[key].tensors
            self.assertEqual(len(x),len(part)-20)
            self.assertEqual(x[0].tolist(),[lookup.get(w,0) for w in part[:20]])
            self.assertEqual(y[0].item(),lookup.get(part[20],0))
            self.assertEqual(y[-1].item(),lookup.get(part[-1],0))
        self.assertNotIn('gutenberg',words); self.assertNotIn('chapter',words)

    def test_model_update_roundtrip_and_generation(self):
        x,y=(t[:4] for t in self.datasets['train'].tensors)
        for kind in e.KINDS:
            with self.subTest(kind=kind):
                e.seed_all(42); model=e.TextModel(kind,len(self.vocab))
                before=model.classifier.weight.detach().clone()
                opt=torch.optim.Adam(model.parameters(),lr=.001)
                logits=model(x); self.assertEqual(tuple(logits.shape),(4,len(self.vocab)))
                loss=torch.nn.functional.cross_entropy(logits,y); self.assertTrue(torch.isfinite(loss))
                loss.backward(); opt.step()
                self.assertFalse(torch.equal(before,model.classifier.weight))
                with tempfile.TemporaryDirectory() as tmp:
                    path=Path(tmp)/'model.pt'; torch.save(model.state_dict(),path)
                    other=e.TextModel(kind,len(self.vocab)); other.load_state_dict(torch.load(path,weights_only=True))
                    self.assertTrue(torch.equal(model(x),other(x)))
                for temp in [.5,1.,1.5]:
                    output=e.generate(model,self.seeds[0],self.vocab,temp,101,torch.device('cpu'))
                    self.assertEqual(len(output),100); self.assertNotIn('<unk>',output)
                self.assertEqual(e.generate(model,self.seeds[0],self.vocab,1.,101,torch.device('cpu')),
                                 e.generate(model,self.seeds[0],self.vocab,1.,101,torch.device('cpu')))

    def test_weighted_evaluation(self):
        model=e.TextModel('GRU',len(self.vocab)); ds=self.datasets['test']
        small=torch.utils.data.TensorDataset(*(t[:19] for t in ds.tensors))
        a=e.evaluate(model,torch.utils.data.DataLoader(small,batch_size=7),torch.device('cpu'))
        b=e.evaluate(model,torch.utils.data.DataLoader(small,batch_size=19),torch.device('cpu'))
        self.assertAlmostEqual(a['loss'],b['loss'],places=5)
        self.assertEqual(a['accuracy'],b['accuracy'])

if __name__=='__main__': unittest.main()
