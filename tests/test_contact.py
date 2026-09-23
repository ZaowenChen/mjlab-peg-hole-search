import unittest
from dataclasses import replace
import numpy as np
import torch
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.controller import ContactController
from mjlab_contact_prep.execution import COLUMNS
from mjlab_contact_prep.metrics import assess


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.c=ContactController(ContactConfig(),2,'cpu')
        self.p=torch.zeros(2,3);self.r=torch.eye(3).repeat(2,1,1)
        self.a=torch.zeros(2,2)

    def run_force(self,force,n):
        w=torch.zeros(2,6);w[:,2]=force
        for _ in range(n):self.c.step(w,w,self.p,self.r,self.a)

    def test_twenty_N_allows_handoff_and_ramp_continuity(self):
        self.run_force(6,50)
        self.assertTrue((self.c.state==self.c.REGULATE).all())
        self.assertTrue((self.c.target<20).all())
        self.run_force(20,300)
        self.assertTrue(self.c.ready.all())
        self.a[:,0]=1
        self.run_force(20,1)
        self.assertTrue((self.c.last_xy.norm(dim=-1)>0).all())

    def test_partial_reset_does_not_clear_other_world(self):
        self.run_force(20,300)
        t=self.c.elapsed[1].item();self.c.reset(torch.tensor([0]))
        self.assertEqual(self.c.elapsed[1].item(),t)
        self.assertEqual(self.c.state[0].item(),self.c.APPROACH)

    def test_fault_latches_and_blocks_lateral(self):
        self.run_force(6,60);self.a[:]=1
        self.run_force(41,1)
        self.assertTrue((self.c.state==self.c.FAULT).all())
        self.assertFalse(self.c.ready.any())
        self.run_force(20,20)
        self.assertTrue((self.c.state==self.c.FAULT).all())
        self.assertTrue((self.c.last_xy==0).all())

    def test_loss_is_bounded_recovery(self):
        self.run_force(20,300);self.run_force(0,100)
        self.assertTrue((self.c.state==self.c.ACQUIRE).all())
        self.run_force(0,600)
        self.assertTrue((self.c.reason==7).all())

    def test_nonfinite_force_stops_motion(self):
        self.run_force(float('nan'),1)
        self.assertTrue((self.c.reason==4).all())
        self.assertTrue((self.c.last_twist==0).all())


class MetricsTests(unittest.TestCase):
    def trace(self):
        a=np.zeros((1600,len(COLUMNS)),dtype=float)
        a[:,COLUMNS.index('time')]=np.arange(len(a))*.002
        for key,val in [('Fz',20),('filtered_force',20),('target',20),('state',2)]:a[:,COLUMNS.index(key)]=val
        return a

    def test_stable_force_requires_contiguous_two_seconds(self):
        a=self.trace();c=ContactConfig()
        r=assess(a,COLUMNS,c)
        self.assertTrue(r['sustained_2s']);self.assertAlmostEqual(r['stable_contact_s'],.248)
        a[::100,COLUMNS.index('reason')]=1
        self.assertFalse(assess(a,COLUMNS,c)['sustained_2s'])

    def test_immediate_overload_is_still_physical_contact(self):
        a=self.trace();a[:,COLUMNS.index('Fz')]=45
        a[:,COLUMNS.index('state')]=3;a[:,COLUMNS.index('reason')]=1
        a[:,COLUMNS.index('target')]=0
        r=assess(a,COLUMNS,ContactConfig())
        self.assertEqual(r['first_touch_s'],0.)
        self.assertFalse(r['sustained_2s'])

    def test_late_instability_is_reported(self):
        a=self.trace();a[-200:,COLUMNS.index('Fz')]=30
        r=assess(a,COLUMNS,ContactConfig())
        self.assertTrue(r['sustained_2s'])
        self.assertFalse(r['stable_through_end'])
        self.assertLess(r['post_confirmation_stable_fraction'],1.)

    def test_filtered_signal_cannot_hide_bad_raw_force(self):
        a=self.trace();a[::2,COLUMNS.index('Fz')]=0;a[1::2,COLUMNS.index('Fz')]=40
        self.assertFalse(assess(a,COLUMNS,ContactConfig())['sustained_2s'])


if __name__=='__main__':unittest.main()
