import unittest
import torch
from mjlab_contact_prep.probe import ConicalReference,ProbeConfig,rotation_exp

class ProbeTests(unittest.TestCase):
    def test_contact_gate_loss_ramp_and_partial_reset(self):
        p=ConicalReference(ProbeConfig(),2,'cpu',.002)
        mean=torch.eye(3).repeat(2,1,1)
        off=torch.zeros(2,dtype=torch.bool);on=~off
        for _ in range(200):p.step(mean,off,off)
        self.assertFalse(p.started.any());self.assertEqual(p.amplitude.sum(),0)
        for _ in range(300):p.step(mean,on,on)
        self.assertTrue(p.started.all());self.assertTrue((p.amplitude>0).all())
        target=p.target[1].clone();phase=p.phase[1].clone()
        p.reset(torch.tensor([0]))
        self.assertTrue(torch.equal(p.target[1],target));self.assertEqual(p.phase[1],phase)
        for _ in range(200):p.step(mean,off,off)
        self.assertEqual(p.amplitude.sum(),0)
        self.assertTrue(torch.allclose(p.target,mean))

    def test_rotation_and_feedforward_follow_commanded_cone(self):
        p=ConicalReference(ProbeConfig(amplitude_deg=.25,frequency=.5,ramp_time=.3),1,'cpu',.002)
        mean=rotation_exp(torch.tensor([[.1,.2,.3]]));on=torch.ones(1,dtype=torch.bool)
        for _ in range(500):target,w,_=p.step(mean,on,on)
        self.assertTrue(torch.allclose(target.transpose(1,2)@target,torch.eye(3)[None],atol=1e-6))
        # Float32 matrix differencing at 2 ms incurs cancellation at this scale.
        self.assertAlmostEqual(float(w.norm()),.0137078,delta=2e-5)
        zero=ConicalReference(ProbeConfig(amplitude_deg=0),1,'cpu',.002)
        for _ in range(500):target,w,_=zero.step(mean,on,on)
        self.assertTrue(torch.allclose(target,mean));self.assertLess(float(w.norm()),1e-5)

    def test_overload_holds_reference_without_reversing_tilt(self):
        p=ConicalReference(ProbeConfig(),1,'cpu',.002)
        mean=torch.eye(3)[None];on=torch.ones(1,dtype=torch.bool);off=~on
        for _ in range(800):p.step(mean,on,on,on)
        target=p.target.clone();phase=p.phase.clone()
        for _ in range(100):p.step(mean,off,off,on)
        self.assertTrue(torch.equal(p.target,target));self.assertTrue(torch.equal(p.phase,phase))

if __name__=='__main__':unittest.main()
