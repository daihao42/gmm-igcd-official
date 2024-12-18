
from utils import dataloader
from utils.dataloader import ClassIncrementalLoader

class TinyImageNetLoader():
    def __init__(self, args):
        self.args = args

    def makeClassIncrementalLoader(self):
        loader = ClassIncrementalLoader(data_dir=self.args.data_dir, pretrained_model_name= self.args.pretrained_model_name, base=self.args.base, increment=self.args.increment)
        train_loader = loader.train_dataloader()

        test_all_loader = loader.test_dataloader(mode='all')
        test_novel_loader = loader.test_dataloader(mode='novel')
        test_old_loader = loader.test_dataloader(mode='old')

        return train_loader, test_novel_loader, test_old_loader, test_all_loader

    # metaGCD-comparison
    def makePromtCCDLoader(self):
        """
        @2024-11-22 to-do : read promtCCD paper and implement the loader
        """
        base = 70
        increment = 10
        num_labeled = 32000
        num_novel_per_stage = 1500
        num_known_per_stage = 2000

        loader = dataloader.StrictClassInstanceIncrementalLoader(data_dir=self.args.data_dir,pretrained_model_name=self.args.pretrained_model_name, base=base, increment=increment, num_labeled=num_labeled, num_novel_per_stage=num_novel_per_stage, num_known_per_stage=num_known_per_stage)

        train_loader = loader.train_dataloader()
        test_all_loader = loader.test_dataloader(mode='all')
        test_novel_loader = loader.test_dataloader(mode='novel')
        test_old_loader = loader.test_dataloader(mode='old')
        
        return train_loader, test_novel_loader, test_old_loader, test_all_loader
    
    # metaGCD-comparison
    def makeMetaGCDLoader(self):
        base = 80
        increment = 5
        num_labeled = 32000
        num_novel_per_stage = 1500
        num_known_per_stage = 2000

        loader = dataloader.StrictClassInstanceIncrementalLoader(data_dir=self.args.data_dir,pretrained_model_name=self.args.pretrained_model_name, base=base, increment=increment, num_labeled=num_labeled, num_novel_per_stage=num_novel_per_stage, num_known_per_stage=num_known_per_stage)

        train_loader = loader.train_dataloader()
        test_all_loader = loader.test_dataloader(mode='all')
        test_novel_loader = loader.test_dataloader(mode='novel')
        test_old_loader = loader.test_dataloader(mode='old')
        
        return train_loader, test_novel_loader, test_old_loader, test_all_loader

    # metaGCD-comparison
    def makeHappyLoader(self):
        base = 100
        increment = 20
        num_labeled = 40000
        num_novel_inc = 400
        num_known_inc = 25

        loader = dataloader.StrictPerClassIncrementalLoader(data_dir=self.args.data_dir, pretrained_model_name=self.args.pretrained_model_name, base=base, increment=increment, num_labeled=num_labeled, num_novel_inc=num_novel_inc, num_known_inc=num_known_inc)

        train_loader = loader.train_dataloader()
        test_all_loader = loader.test_dataloader(mode='all')
        test_novel_loader = loader.test_dataloader(mode='novel')
        test_old_loader = loader.test_dataloader(mode='old')
        
        return train_loader, test_novel_loader, test_old_loader, test_all_loader

