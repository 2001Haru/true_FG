import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_source_dataset import (PairedEpochSampler, PairedHardDataset,
                                   PairedSoftLoadDataset, PairedSoftSaveDataset)


class PairedSourceDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); root = Path(self.temp.name)
        ref0 = root / "red.png"; ref1 = root / "blue.png"; packed = root / "packed.png"
        Image.new("RGB", (224, 224), (255, 0, 0)).save(ref0)
        Image.new("RGB", (224, 224), (0, 0, 255)).save(ref1)
        image = Image.new("RGB", (224, 224)); image.paste(Image.new("RGB", (224, 112), (0, 255, 0)), (0, 0)); image.paste(Image.new("RGB", (224, 112), (0, 0, 255)), (0, 112)); image.save(packed)
        rows=[]
        for parent in range(300):
            rows.append({"parent_index":parent,"class_id":parent//3,"class":f"{parent//3:03d}","slot":parent%3,
                         "packed_path":str(packed),"sources":[
                             {"reference_path":str(ref0),"density":[.5]*224,"identity":"red","official_bbox_1indexed":[1,57,224,168],"raw_size":[224,224]},
                             {"reference_path":str(ref1),"density":[.5]*224,"identity":"blue","official_bbox_1indexed":[1,57,224,168],"raw_size":[224,224]}]})
        manifest=root/"manifest.json";manifest.write_text(json.dumps({"status":"complete","parents":300,"classes":[f"{i:03d}" for i in range(100)],"schedule_seed":42,"records":rows}));self.manifest=manifest;self.root=root

    def tearDown(self): self.temp.cleanup()

    def test_persistent_workers_observe_epoch_and_balance_sources(self):
        dataset=PairedSoftSaveDataset(self.manifest);sampler=PairedEpochSampler(dataset,42)
        loader=DataLoader(dataset,batch_size=20,sampler=sampler,num_workers=2,persistent_workers=True)
        counts=np.zeros((300,2),dtype=int)
        for epoch in range(4):
            dataset.set_epoch(epoch)
            for batch in loader:
                for parent,source in zip(batch[6].tolist(),batch[5].tolist()): counts[parent,source]+=1
        self.assertTrue(np.all(counts==2))

    def test_load_uses_cached_source_and_parent(self):
        fkd=self.root/"fkd"; (fkd/"epoch_0").mkdir(parents=True)
        (fkd/"relabel_manifest.json").write_text('{}')
        dataset=PairedSoftLoadDataset(self.manifest,"reference",fkd,1,20,42);dataset.set_epoch(0);order=dataset.sampler.permutation(0)
        for batch in range(15):
            parents=torch.tensor(order[batch*20:(batch+1)*20]);sources=torch.ones(20,dtype=torch.long)
            coords=torch.tensor([[0.,0.,1.,1.]]*20);flip=torch.zeros(20,dtype=torch.bool)
            torch.save([coords,flip,torch.arange(20),1.,[0,0,0,0],torch.zeros(20,100),sources,parents],fkd/f"epoch_0/batch_{batch}.tar")
        first=(order[0],0);dataset.load_batch_config(first);image,_,_,_=dataset[first]
        self.assertGreater(float(image[2].mean()),float(image[0].mean()))

    def test_hard_schedule_is_exactly_balanced(self):
        dataset=PairedHardDataset(self.manifest,"reference",42);audit=dataset.trajectory_audit(600)
        self.assertEqual(audit["examples"],180000);self.assertEqual(audit["source_exposure_min"],300);self.assertEqual(audit["source_exposure_max"],300)

    def test_hybrid_modes_split_object_and_background_rows(self):
        from paired_source_dataset import PairedSourceIndex
        object_view=np.asarray(PairedSourceIndex(self.manifest,"object_decoded").load(0,0))
        background_view=np.asarray(PairedSourceIndex(self.manifest,"background_decoded").load(0,0))
        smooth_view=np.asarray(PairedSourceIndex(self.manifest,"background_hsmooth").load(0,0))
        self.assertTrue(np.all(object_view[80]==np.array([0,255,0])));self.assertTrue(np.all(object_view[20]==np.array([255,0,0])))
        self.assertTrue(np.all(background_view[80]==np.array([255,0,0])));self.assertTrue(np.all(background_view[20]==np.array([0,255,0])))
        self.assertTrue(np.all(smooth_view[80]==np.array([0,255,0])));self.assertTrue(np.all(smooth_view[220]==np.array([0,255,0])))


if __name__ == "__main__": unittest.main()
