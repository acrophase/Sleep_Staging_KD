from .mass import MassDataModule
from .mass_batch import MASSBatchDataModule
from .mass_batch_small import MASSBatchSmallDataModule
from .mass_iter import MASSIterDataModule

available_datasets = {
                        "mass": MassDataModule,
                        "mass_batch": MASSBatchDataModule,
                        "mass_batch_small": MASSBatchSmallDataModule,
                        "mass_iter": MASSIterDataModule
                        }