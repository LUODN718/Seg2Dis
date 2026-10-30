import tempfile
from pathlib import Path
import unittest
import torch
from models import VARIANTS,build_model
from train import compute_loss,make_optimizer
from utils.checkpoint import model_metadata,validate_checkpoint,load_model

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(42)
        cls.model=build_model(backbone_checkpoint='').eval()
        cls.image=torch.randn(1,3,64,64)

    def test_complete_structure(self):
        model=self.model
        self.assertEqual(sum(p.numel() for p in model.parameters()),28376678)
        self.assertEqual([b.fusion[0][0].in_channels for b in model.foreground.scale_blocks],[290]*3)
        self.assertEqual(model.foreground.boundary_block.smooth_body[0][0].in_channels,291)
        self.assertEqual(model.foreground.boundary_block.sharp_body[0][0].in_channels,291)
        for name,_ in model.named_modules():
            self.assertFalse(any(x in name.lower() for x in ('frequency','haar','wavelet')))
        self.assertFalse(any('frequency' in k.lower() or 'haar' in k.lower() for k in model.state_dict()))

    def test_composition_and_current_input(self):
        with torch.inference_mode():out=self.model(self.image)
        self.assertEqual(out['three_class_probability'].shape,(1,3,64,64))
        torch.testing.assert_close(out['three_class_probability'].sum(1),torch.ones(1,64,64))
        torch.testing.assert_close(out['three_class_probability'][:,:1],1-out['Pu'])
        torch.testing.assert_close(out['three_class_probability'][:,1:],out['Pu']*out['type_probability'])
        torch.testing.assert_close(out['P0'],out['base_final'].sigmoid())
        self.assertTrue(torch.equal(out['residual_input'][:,:3],self.image))
        self.assertEqual(out['residual_input'].shape[1],5)

    def test_pm_rgb_u_input_and_strict_checkpoint_reload(self):
        model=build_model(backbone_checkpoint='',residual_input_mode='rgb_pm_u').eval()
        with torch.inference_mode():out=model(self.image)
        self.assertEqual(out['residual_input'].shape[1],5)
        torch.testing.assert_close(out['residual_input'][:,:3],self.image)
        torch.testing.assert_close(out['residual_input'][:,3:4],out['master_final'].sigmoid())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'pm_rgb_u.pth'
            torch.save({**model_metadata(model),'model_state':model.state_dict(),
                        'image_size':352,'run_kind':'smoke_test','epoch':0},path)
            restored,_,_=load_model(path,'cpu')
            self.assertEqual(restored.foreground.residual_input_mode,'rgb_pm_u')
            with torch.inference_mode():actual=restored(self.image)['three_class_probability']
        torch.testing.assert_close(actual,out['three_class_probability'],rtol=0,atol=0)

    def test_all_variants_forward_backward(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(42)
                model=build_model(backbone_checkpoint='',variant=variant).train()
                batch={'image':self.image,'semantic':torch.randint(0,3,(1,64,64))}
                loss,_=compute_loss(model,batch,torch.device('cpu'))
                self.assertTrue(torch.isfinite(loss));loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertIsNotNone(model.foreground.backbone.patch_embed.proj.weight.grad)
                if variant=='flr':
                    self.assertFalse(hasattr(model,'query'))
                    with torch.no_grad():out=model(self.image)
                    self.assertTrue(torch.equal(out['P0'],out['Pu']))
                    self.assertNotIn('three_class_probability',out)
                if variant=='segdis_no_correction':
                    self.assertFalse(hasattr(model,'correction'))
                    with torch.no_grad():out=model(self.image)
                    self.assertTrue(torch.equal(out['P0'],out['Pu']))
                if variant=='segdis_no_rta':
                    self.assertFalse(hasattr(model,'query_condition'))
                    self.assertFalse(hasattr(model,'correction'))
                    self.assertFalse(model.relation_conditioning)
                    with torch.no_grad():out=model(self.image)
                    self.assertTrue(torch.equal(out['P0'],out['Pu']))
                del model

    def test_optimizer_step_checkpoint_and_strict_reload(self):
        model=self.model.train()
        config={'backbone_lr':1e-4,'base_lr':1e-4,'weight_decay':5e-4}
        optimizer=make_optimizer(model,config)
        batch={'image':self.image,'semantic':torch.randint(0,3,(1,64,64))}
        optimizer.zero_grad(set_to_none=True)
        loss,_=compute_loss(model,batch,torch.device('cpu'));loss.backward();optimizer.step()
        optimizer.zero_grad(set_to_none=True);model.eval()
        with torch.no_grad():reference=model(self.image)['three_class_probability']
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.pth'
            torch.save({**model_metadata(model),'model_state':model.state_dict(),
                        'image_size':352,'run_kind':'smoke_test','epoch':0},path)
            restored,state,identity=load_model(path,'cpu')
            self.assertEqual(identity['run_kind'],'smoke_test')
            with torch.no_grad():actual=restored(self.image)['three_class_probability']
            torch.testing.assert_close(reference,actual,rtol=0,atol=0)
            state['model_variant']='flr'
            with self.assertRaises(ValueError):validate_checkpoint(state,model)

    def test_reject_unidentified_or_tampered_weights(self):
        with self.assertRaises(ValueError):validate_checkpoint({'model_state':self.model.state_dict()})
        state={**model_metadata(self.model),'model_state':dict(self.model.state_dict())}
        validate_checkpoint(state,self.model)
        state['model_state']['foreground.scale_blocks.0.fusion.0.0.weight']=torch.zeros(96,386,3,3)
        with self.assertRaises(ValueError):validate_checkpoint(state,self.model)

if __name__=='__main__':unittest.main()
