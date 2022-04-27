from .mass import MassDataModule
from .mass_batch import MASSBatchDataModule
from .mass_batch_small import MASSBatchSmallDataModule

available_datasets = {
                        "mass": MassDataModule,
                        "mass_batch": MASSBatchDataModule,
                        "mass_batch_small": MASSBatchSmallDataModule
                        }