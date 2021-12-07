from .mass import MassDataModule
from .mass_5min import Mass_5min_DataModule

available_datasets = {
                        "mass": MassDataModule,
                        "mass_5min": Mass_5min_DataModule,
                        }